import os
import gc
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from mmpretrain import get_model
import torchvision.models as models
import einops

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
import sys
sys.path.append('insightface/recognition/arcface_torch/')
import cv2
from insightface.app import FaceAnalysis




from backbones.iresnet import iresnet100
import torch

import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats

from pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import parse_args_paired_training, PairedDataset, CustomData, VideoData

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # VGG expected mean
                         std=[0.229, 0.224, 0.225]),
])

def gram_matrix(feat):
    b, c, h, w = feat.shape
    feat = feat.view(b, c, h * w)
    gram = torch.bmm(feat, feat.transpose(1, 2))  # [B, C, C]
    return gram / (c * h * w)  # normalized

class VGGStyleFeatures(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.vgg = models.vgg16(pretrained=True).features[:max(layers)+1].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.layers = layers

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                feats.append(x)
        return feats

def crop_face_affine(input_tensor, bbox, output_size=(112, 112)):
    """
    input_tensor: [B, C, H, W]
    bbox: torch.tensor([x1, y1, x2, y2]) normalized to [0, 1]
    output_size: desired face crop size
    """
    B, C, H, W = input_tensor.shape
    x1, y1, x2, y2 = bbox  # assume normalized
    theta = torch.tensor([[
        [(x2 - x1), 0, x1 + (x2 - x1)/2 - 0.5],
        [0, (y2 - y1), y1 + (y2 - y1)/2 - 0.5]
    ]], dtype=torch.float, device=input_tensor.device)  # [1, 2, 3]

    grid = F.affine_grid(theta, size=(B, C, *output_size), align_corners=False)
    face_crop = F.grid_sample(input_tensor, grid, align_corners=False)
    return face_crop

def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir='runs/'
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    if args.pretrained_model_name_or_path == "stabilityai/sd-turbo":
        net_pix2pix = Pix2Pix_Turbo(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae, update_forward=True)
        net_pix2pix.set_train()

    elif args.pretrained_model_name_or_path != "":
        if args.pretrained_model_name_or_path[-3:] == 'pkl':
            net_pix2pix = Pix2Pix_Turbo(pretrained_name='', pretrained_path=args.pretrained_model_name_or_path, update_forward=True)
            net_pix2pix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_clip, _ = clip.load("ViT-B/32", device="cuda")
    net_clip.requires_grad_(False)
    net_clip.eval()

    net_lpips.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())
    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    dataset_train = VideoData(image_prep=args.train_image_prep, split="train", tokenizer=net_pix2pix.tokenizer)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = VideoData(image_prep=args.test_image_prep, split="test", tokenizer=net_pix2pix.tokenizer)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=0)

    # Prepare everything with our `accelerator`.
    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_clip, net_lpips = accelerator.prepare(net_clip, net_lpips)
    # renorm with image net statistics
    t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # turn off eff. attn for the discriminator
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # compute the reference stats for FID tracking
    # if accelerator.is_main_process and args.track_val_fid:
    #     feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)

    #     def fn_transform(x):
    #         x_pil = Image.fromarray(x)
    #         out_pil = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS)(x_pil)
    #         return np.array(out_pil)

    #     ref_stats = get_folder_features(args.dataset_folder, model=feat_model, num_workers=0, num=None,
    #             shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
    #             mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)

    # arcloss_fn = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
    # arc_model = get_model('resnet50-arcface_inshop', pretrained=True).cuda()
    # arc_model.eval()
    
    
    # Example style layers from VGG (based on Gatys et al.)
    style_layers = [1, 6, 11, 18, 25]  # conv1_1, conv2_1, ..., conv5_1
    vgg_features = VGGStyleFeatures(style_layers).cuda()
    
    # 1. Detect face once (doesn't need gradients)
    detector = FaceAnalysis(name='buffalo_l')
    detector.prepare(ctx_id=0, det_size=(256,256))
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # arcface_model = iresnet100(pretrained=True)
    arcface_model = iresnet100(pretrained=False).cuda().eval()  # don't use built-in load
    ckpt = torch.load("checkpoints/arcface_backbone.pth", map_location='cpu')
    arcface_model.load_state_dict(ckpt)
    arcface_model = arcface_model.to(dtype=torch.bfloat16)
    
    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_pix2pix, net_disc]
            with accelerator.autocast():
                with accelerator.accumulate(*l_acc):
                    x_src = batch["conditioning_pixel_values"]
                    x_tgt = batch["output_pixel_values"]
                    B, V, C, H, W = x_src.shape


                    if global_step < 1000:
                        l2_weight = 1.0
                        lpips_weight = 0.05
                    elif global_step < 3000:
                        l2_weight = 1.5
                        lpips_weight = 0.1
                    else:
                        l2_weight = 2.0
                        lpips_weight = 0.2
                        
                    arc_weight = 0
                    style_weight = 0   
                    if step > 0:
                        arc_weight = 0.3
                        style_weight = 1.5

                    if step > 5000:
                        arc_weight = 0.4  # identity matters more now
                        style_weight = 2.0

                    if step > 10000:
                        arc_weight = 0.5
                        style_weight = 2.0

                    # forward pass
                    torch.cuda.empty_cache()
                    # print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    # print(f"Reserved : {torch.cuda.memory_reserved()  / 1024**2:.2f} MB")
                    x_tgt_pred = net_pix2pix(x_src, prompt=["" for _ in range(B)], deterministic=True) # prompt_tokens=batch["input_ids"]
                    # Reconstruction loss
                    # loss_l2 = F.mse_loss(x_tgt_pred[:, :-1].reshape(B*(V-1), C, H, W).float(), x_tgt[:, :-1].reshape(B*(V-1), C, H, W).float(), reduction="mean") * args.lambda_l2
                    # loss_lpips = net_lpips(x_tgt_pred[:, :-1].reshape(B*(V-1), C, H, W).float(), x_tgt[:, :-1].reshape(B*(V-1), C, H, W).float()).mean() * args.lambda_lpips
                    
                    loss_l2 = F.mse_loss(x_tgt_pred[:, 1].reshape(B, C, H, W).float(), x_tgt[:, 1].reshape(B, C, H, W).float(), reduction="mean")  
                    loss_lpips = net_lpips(x_tgt_pred[:, 1].reshape(B, C, H, W).float(), x_tgt[:, 1].reshape(B, C, H, W).float()).mean() 
                    
                    # diff = F.mse_loss(x_tgt[:, 0].reshape(B, C, H, W).float(), x_tgt[:, 1].reshape(B, C, H, W).float(), reduction="mean") 
                    # diff_pred = F.mse_loss(x_tgt_pred[:, 0].reshape(B, C, H, W).float(), x_tgt[:, 1].reshape(B, C, H, W).float(), reduction="mean") 
                    # loss_temporal = torch.abs(diff - diff_pred)
                    
                    with torch.no_grad():
                        tgt = img_transform(x_tgt[:,1])
                        pred = img_transform(x_tgt_pred[:,1])
                        tgt_feats   = vgg_features(tgt)
                        pred_feats = vgg_features(pred)
                        style_loss = 0.0
                        for sf, cf in zip(tgt_feats, pred_feats):
                            Gs = gram_matrix(sf)
                            Gc = gram_matrix(cf)
                            style_loss += torch.nn.functional.mse_loss(Gs, Gc) 
                            
                        
                        # crop_tgt = []
                        # crop_pred = []
                        # for b in range(x_tgt.shape[0]):
                        #     x_tgt_np = einops.rearrange(x_tgt[b,1].float().detach().cpu().numpy(), 'c h w -> h w c')
                        #     x_tgt_np = ((x_tgt_np + 1) * 127.5).astype(np.uint8)

                        #     face_tgt = detector.get(x_tgt_np)
                        #     x_tgt_np = einops.rearrange(x_tgt_pred[b,1].float().detach().cpu().numpy(), 'c h w -> h w c')
                        #     x_tgt_np = ((x_tgt_np + 1) * 127.5).astype(np.uint8)
                            
                        #     face_pred = detector.get(x_tgt_np)
                            
                            # if len(face_tgt) == 0 or len(face_pred) == 0:
                            #     continue
                            #     # loss_arc += torch.tensor(0.0, requires_grad=True).cuda()
                            # else:
                            #     bbox = face_tgt[0].bbox 
                            #     H_1, W_1 = x_tgt.shape[-2:]
                            #     x1, y1, x2, y2 = bbox
                            #     bbox_norm = torch.tensor([x1/W_1, y1/H_1, x2/W_1, y2/H_1]).cuda()
                            #     face_crop = crop_face_affine(x_tgt[[b],1].float(), bbox_norm)
                            #     face_crop_tgt = face_crop.to(dtype=torch.bfloat16)    
                            #     crop_tgt += [face_crop_tgt]
                            #     # target_feat = arcface_model(face_crop_tgt)
                                                                
                            #     bbox = face_pred[0].bbox 
                            #     H_1, W_1 = x_tgt_pred.shape[-2:]
                            #     x1, y1, x2, y2 = bbox
                            #     bbox_norm = torch.tensor([x1/W_1, y1/H_1, x2/W_1, y2/H_1]).cuda()
                            #     face_crop = crop_face_affine(x_tgt_pred[[b],1].float(), bbox_norm)
                            #     face_crop_pred = face_crop.to(dtype=torch.bfloat16)
                            #     crop_pred += [face_crop_pred]
                                                        
                                # pred_feat = arcface_model(face_crop_pred)
                        bbox_norm = [0.35, 0.10, 0.65, 0.40]
                        face_crop = crop_face_affine(x_tgt_pred[:,1].float(), bbox_norm)
                        face_crop_pred = face_crop.to(dtype=torch.bfloat16)
                        face_crop = crop_face_affine(x_tgt[:,1].float(), bbox_norm)
                        face_crop_tgt = face_crop.to(dtype=torch.bfloat16) * torch.bernoulli(torch.full((B, 1, 1, 1), 0.9)).cuda().to(dtype=torch.bfloat16)
                        # crop_tgt = torch.cat(crop_tgt, axis=0)
                        # crop_pred = torch.cat(crop_pred, axis=0)
                        target_feat = arcface_model(face_crop_tgt)
                        pred_feat = arcface_model(face_crop_pred)
                        loss_arc = 1 - F.cosine_similarity(pred_feat, target_feat, dim=1).mean() 
                        loss_face = net_lpips(face_crop_tgt, face_crop_pred).mean()
                        # else:
                        #     loss_arc = torch.tensor(0.0, requires_grad=True).cuda() 
                            # loss_face = net_lpips(face_crop_tgt, face_crop_pred).mean() * args.lambda_lpips
                        # print(target_feat)
                    
                    # anchor = torch.ones(B, requires_grad=True).cuda()
                    # pred_arc = arc_model(x_tgt_pred[:,0,:,:])[0]
                    # tgt_arc = arc_model(x_tgt[:,0,:,:])[0]
                    # pred_arc1 = arc_model(x_tgt_pred[:,1,:,:])[0]
                    # tgt_arc1 = arc_model(x_tgt[:,1,:,:])[0]
                    # loss_arc = (arcloss_fn(pred_arc.float(), tgt_arc.float(), anchor.float()) + arcloss_fn(pred_arc1.float(), tgt_arc1.float(), anchor.float())) * 10
                    # if step < 1000:
                    #     loss = loss_l2 #+ loss_lpips + style_loss + loss_arc #+ loss_face #+ loss_arc
                    # else:
                    loss = loss_l2 * l2_weight + loss_lpips * lpips_weight + style_loss * style_weight + loss_arc * arc_weight + loss_face * lpips_weight

                    if global_step % args.eval_freq == 1:
                        os.makedirs(os.path.join(args.output_dir, "train", f"fid_{global_step}"), exist_ok=True)
                        output_img = torch.cat([x_tgt[0][2], x_tgt[0][0], x_tgt[0][1], x_src[0][0], x_src[0][1], x_tgt_pred[0][0], x_tgt_pred[0][1]], dim=2)
                        # output_img = torch.cat([x_tgt[0][1], x_tgt[0][0], x_src[0][1], x_src[0][0], x_tgt_pred[0][1], x_tgt_pred[0][0]], dim=2)
                        output_pil = transforms.ToPILImage()(output_img.cpu() * 0.5 + 0.5)
                        outf = os.path.join(args.output_dir, "train", f"fid_{global_step}", f"val_{step}.png")
                        output_pil.save(outf)
                    # CLIP similarity loss
                    # if args.lambda_clipsim > 0:
                    #     x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5).reshape(B*V, C, H, W)
                    #     x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                    #     caption_tokens = clip.tokenize(batch["caption"], truncate=True).to(x_tgt_pred.device)
                    #     clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                    #     loss_clipsim = (1 - clipsim.mean() / 100)
                    #     loss += loss_clipsim * args.lambda_clipsim
                    accelerator.backward(loss, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    """
                    Generator loss: fool the discriminator
                    """
                    x_tgt_pred = net_pix2pix(x_src, prompt=["" for _ in range(B)], deterministic=True)
                    lossG = net_disc(x_tgt_pred[:,1].reshape(B, C, H, W), for_G=True).mean() * args.lambda_gan
                    accelerator.backward(lossG)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    """
                    Discriminator loss: fake image vs real image
                    """
                    # real image
                    lossD_real = net_disc(x_tgt[:,1].detach().reshape(B, C, H, W), for_real=True).mean() * args.lambda_gan
                    accelerator.backward(lossD_real.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    # fake image
                    lossD_fake = net_disc(x_tgt_pred[:,1].detach().reshape(B, C, H, W), for_real=False).mean() * args.lambda_gan
                    accelerator.backward(lossD_fake.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    lossD = lossD_real + lossD_fake

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        logs = {}
                        # log all the losses
                        logs["lossG"] = lossG.detach().item()
                        logs["lossD"] = lossD.detach().item()
                        logs["loss_l2"] = loss_l2.detach().item()
                        logs["loss_lpips"] = loss_lpips.detach().item()
                        logs["loss_style"] = style_loss.detach().item()
                        logs["loss_arc"] = loss_arc.detach().item()
                        logs["loss_face"] = loss_face.detach().item()
                        # if args.lambda_clipsim > 0:
                        #     logs["loss_clipsim"] = loss_clipsim.detach().item()
                        progress_bar.set_postfix(**logs)

                        # viz some images
                        # if global_step % args.viz_freq == 1:
                        #     log_dict = {
                        #         "train/source": [wandb.Image(x_src[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        #         "train/target": [wandb.Image(x_tgt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        #         "train/model_output": [wandb.Image(x_tgt_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        #     }
                        #     for k in log_dict:
                        #         logs[k] = log_dict[k]

                        # checkpoint the model
                        if global_step % args.checkpointing_steps == 1:
                            outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                            accelerator.unwrap_model(net_pix2pix).save_model(outf)

                        # compute validation set FID, L2, LPIPS, CLIP-SIM
                        if global_step % args.eval_freq == 1:
                            l_l2, l_lpips, l_arc, l_style, l_face = [], [], [], [], []
                            os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                            for step, batch_val in enumerate(dl_val):
                                if step >= args.num_samples_eval:
                                    break
                                x_src = batch_val["conditioning_pixel_values"].cuda()
                                x_tgt = batch_val["output_pixel_values"].cuda()
                                B, V, C, H, W = x_src.shape
                                assert B == 1, "Use batch size 1 for eval."
                                with torch.no_grad():
                                    # forward pass
                                    x_tgt_pred = accelerator.unwrap_model(net_pix2pix)(x_src, prompt=["" for _ in range(B)], deterministic=True)
                                    # compute the reconstruction losses
                                    loss_l2 = F.mse_loss(x_tgt_pred[:, 1].float(), x_tgt[:, 1].float(), reduction="mean")
                                    loss_lpips = net_lpips(x_tgt_pred[:, 1].float(), x_tgt[:, 1].float()).mean()
                                    # compute clip similarity loss
                                    # x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5).reshape(B*V, C, H, W)
                                    # x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                                    # caption_tokens = clip.tokenize(batch_val["caption"], truncate=True).to(x_tgt_pred.device)
                                    # clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                                    # clipsim = clipsim.mean()

                                    # anchor = torch.ones(B, requires_grad=True).cuda()
                                    # pred_arc = arc_model(x_tgt_pred[:,0,:,:])[0]
                                    # tgt_arc = arc_model(x_tgt[:,0,:,:])[0]
                                    # # loss_arc = arcloss_fn(pred_arc.float(), tgt_arc.float(), anchor.float())

                                    # anchor = torch.ones(B, requires_grad=True).cuda()
                                    # pred_arc = arc_model(x_tgt_pred[:,1,:,:])[0]
                                    # tgt_arc = arc_model(x_tgt[:,1,:,:])[0]
                                    # loss_arc += arcloss_fn(pred_arc.float(), tgt_arc.float(), anchor.float())
                                    
                                    
                                    # diff = F.mse_loss(x_tgt[:, 0].reshape(B, C, H, W).float(), x_tgt[:, 1].reshape(B, C, H, W).float(), reduction="mean") 
                                    # diff_pred = F.mse_loss(x_tgt_pred[:, 0].reshape(B, C, H, W).float(), x_tgt[:, 1].reshape(B, C, H, W).float(), reduction="mean") 
                                    # loss_temporal = torch.abs(diff - diff_pred)
                                    
                                    tgt = img_transform(x_tgt[:,1])
                                    pred = img_transform(x_tgt_pred[:,1])
                                    tgt_feats   = vgg_features(tgt)
                                    pred_feats = vgg_features(pred)
                                    style_loss = 0.0
                                    for sf, cf in zip(tgt_feats, pred_feats):
                                        Gs = gram_matrix(sf)
                                        Gc = gram_matrix(cf)
                                        style_loss += torch.nn.functional.mse_loss(Gs, Gc)
                                        
                                        
                                    # crop_tgt = []
                                    # crop_pred = []
                                    # for b in range(x_tgt.shape[0]):
                                    #     x_tgt_np = einops.rearrange(x_tgt[b,1].float().detach().cpu().numpy(), 'c h w -> h w c')
                                    #     x_tgt_np = ((x_tgt_np + 1) * 127.5).astype(np.uint8)

                                    #     face_tgt = detector.get(x_tgt_np)
                                    #     x_tgt_np = einops.rearrange(x_tgt_pred[b,1].float().detach().cpu().numpy(), 'c h w -> h w c')
                                    #     x_tgt_np = ((x_tgt_np + 1) * 127.5).astype(np.uint8)
                                        
                                    #     face_pred = detector.get(x_tgt_np)
                                        
                                        
                                    #     if len(face_tgt) == 0 or len(face_pred) == 0:
                                    #         continue
                                    #         # loss_arc += torch.tensor(0.0, requires_grad=True).cuda()
                                    #     else:
                                    #         bbox = face_tgt[0].bbox 
                                    #         H_1, W_1 = x_tgt.shape[-2:]
                                    #         x1, y1, x2, y2 = bbox
                                    #         bbox_norm = torch.tensor([x1/W_1, y1/H_1, x2/W_1, y2/H_1]).cuda()
                                    #         face_crop = crop_face_affine(x_tgt[[b],1].float(), bbox_norm)
                                    #         face_crop_tgt = face_crop.to(dtype=torch.bfloat16)    
                                    #         crop_tgt += [face_crop_tgt]
                                    #         # target_feat = arcface_model(face_crop_tgt)
                                                                           
                                    #         bbox = face_pred[0].bbox 
                                    #         H_1, W_1 = x_tgt_pred.shape[-2:]
                                    #         x1, y1, x2, y2 = bbox
                                    #         bbox_norm = torch.tensor([x1/W_1, y1/H_1, x2/W_1, y2/H_1]).cuda()
                                    #         face_crop = crop_face_affine(x_tgt_pred[[b],1].float(), bbox_norm)
                                    #         face_crop_pred = face_crop.to(dtype=torch.bfloat16)
                                    #         crop_pred += [face_crop_pred]
                                                                    
                                    #         # pred_feat = arcface_model(face_crop_pred)
                                    # if len(crop_tgt) != 0:
                                    #     crop_tgt = torch.cat(crop_tgt, axis=0)
                                    #     crop_pred = torch.cat(crop_pred, axis=0)
                                    #     target_feat = arcface_model(crop_tgt)
                                    #     pred_feat = arcface_model(crop_pred)
                                    #     loss_arc = 1 - F.cosine_similarity(pred_feat, target_feat, dim=1).mean()
                                    # else:
                                    #     loss_arc = torch.tensor(0.0, requires_grad=True).cuda()
                                    bbox_norm = [0.4, 0.15, 0.6, 0.35]
                                    face_crop = crop_face_affine(x_tgt_pred[:,1].float(), bbox_norm)
                                    face_crop_pred = face_crop.to(dtype=torch.bfloat16) 
                                    face_crop = crop_face_affine(x_tgt[:,1].float(), bbox_norm)
                                    face_crop_tgt = face_crop.to(dtype=torch.bfloat16) * torch.bernoulli(torch.full((B, 1, 1, 1), 0.9)).cuda().to(dtype=torch.bfloat16)
                                    # crop_tgt = torch.cat(crop_tgt, axis=0)
                                    # crop_pred = torch.cat(crop_pred, axis=0)
                                    target_feat = arcface_model(face_crop_tgt)
                                    pred_feat = arcface_model(face_crop_pred)
                                    loss_arc = 1 - F.cosine_similarity(pred_feat, target_feat, dim=1).mean() 
                                    loss_face = net_lpips(face_crop_tgt, face_crop_pred).mean() 
                                    
                                    l_l2.append(loss_l2.item())
                                    l_lpips.append(loss_lpips.item())
                                    l_style.append(style_loss.item())
                                    l_arc.append(loss_arc.item())
                                    l_face.append(loss_face.item())
                                    # l_clipsim.append(clipsim.item())
                                # save output images to file for FID evaluation
                                output_img = torch.cat([x_tgt[0][1], x_src[0][1], x_tgt_pred[0][1]], dim=2)
                                # output_img = torch.cat([x_tgt[0][2], x_tgt[0][0], x_tgt[0][1], x_src[0][0], x_src[0][1], x_tgt_pred[0][0], x_tgt_pred[0][1]], dim=2)
                                output_pil = transforms.ToPILImage()(output_img.cpu() * 0.5 + 0.5)
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"val_{step}.png")
                                output_pil.save(outf)
                                # outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"val_{step}.png")
                                # output_pil.save(outf)
                                # output_pil = transforms.ToPILImage()(x_tgt_pred[0][1].cpu() * 0.5 + 0.5)
                                # outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"val_{step}-ref.png")
                                # output_pil.save(outf)
                            # if args.track_val_fid:
                            #     curr_stats = get_folder_features(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), model=feat_model, num_workers=0, num=None,
                            #             shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                            #             mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)
                            #     fid_score = fid_from_feats(ref_stats, curr_stats)
                            #     logs["val/clean_fid"] = fid_score
                            logs["val/l2"] = np.mean(l_l2)
                            logs["val/lpips"] = np.mean(l_lpips)
                            logs["val/style"] = np.mean(l_style)
                            logs["val/loss_arc"] = np.mean(l_arc)
                            logs["val/loss_face"] = np.mean(l_face)
                            gc.collect()
                            torch.cuda.empty_cache()
                        accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
