import os
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from glob import glob
from torchvision.transforms import Resize, Pad
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
import einops
import cv2


def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=3, type=float)
    parser.add_argument("--lambda_l2", default=3.0, type=float)
    parser.add_argument("--lambda_clipsim", default=5.0, type=float)

    # dataset options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=10, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=256,)
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def parse_args_unpaired_training():
    """
    Parses command-line arguments used for configuring an unpaired session (CycleGAN-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """

    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")

    # fixed random seed
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid")
    parser.add_argument("--lambda_gan", default=0.03, type=float)
    parser.add_argument("--lambda_idt", default=1, type=float)
    parser.add_argument("--lambda_cycle", default=1, type=float)
    parser.add_argument("--lambda_cycle_lpips", default=10.0, type=float)
    parser.add_argument("--lambda_idt_lpips", default=1.0, type=float)

    # args for dataset and dataloader options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_img_prep", required=True)
    parser.add_argument("--val_img_prep", required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)

    # args for the model
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sd-turbo")
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument("--variant", default=None, type=str)
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # args for validation and logging
    parser.add_argument("--viz_freq", type=int, default=20)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str, required=True)
    parser.add_argument("--validation_steps", type=int, default=500,)
    parser.add_argument("--validation_num_images", type=int, default=-1, help="Number of images to use for validation. -1 to use all images.")
    parser.add_argument("--checkpointing_steps", type=int, default=500)

    # args for the optimization options
    parser.add_argument("--learning_rate", type=float, default=1e-5,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # memory saving options
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--gradient_checkpointing", action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")

    args = parser.parse_args()
    return args


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T

class ITWData(torch.utils.data.Dataset):
    def __init__(self, split, image_prep, tokenizer):
        self.frame_list = []
        self.root_dir = '../qualitative-eval-data/lady4/gp-avatar/'
        frames = os.listdir(self.root_dir)
        frames.sort()
        self.img_resize = Resize((256,256))
        self.tokenizer = tokenizer
        frames_i = [(frames[i*2], frames[i*2 +1]) for i in range(len(frames)//2)]
        # frames_i += [(frames[6], frames[0])]
        self.frame_list.extend(frames_i)
    
    def __len__(self):
        return len(self.frame_list)
    
    def __getitem__(self, idx):
            
        path_1, path_2 = self.frame_list[idx][0], self.frame_list[idx][1]
        
        img1 = Image.open(self.root_dir + path_1)
        img2 = Image.open(self.root_dir + path_2)
        
        # if self.frame_list[idx][0][0] != 'n':
        #     img1 = np.array(img1)
        #     img1 = linear2color_corr(img1)
        #     img1 = einops.rearrange(torch.tensor(img1), 'h w c -> c h w')
        #     img2 = np.array(img2)
        #     img2 = linear2color_corr(img2)
        #     img2 = einops.rearrange(torch.tensor(img2), 'h w c -> c h w')
        # else:
        img1 = pil_to_tensor(img1)
        img2 = pil_to_tensor(img2)
        ref = img1[:,:, :512]
        ref = self.img_resize(ref)
        ref = (ref / 127.5) - 1.0
        
        noisy_1 = img1[:,:,512:1024]
        noisy_1 = self.img_resize(noisy_1)
        noisy_1 = (noisy_1 / 127.5) - 1.0

        noisy_2 = img2[:,:,512:1024]
        noisy_2 = self.img_resize(noisy_2)
        noisy_2 = (noisy_2 / 127.5) - 1.0
        
        target_1 = img1[:,:,512:1024]
        target_1 = self.img_resize(target_1)
        target_1 = (target_1 / 127.5) - 1.0
        
        target_2 = img2[:,:,512:1024]
        target_2 = self.img_resize(target_2)
        target_2 = (target_2 / 127.5) - 1.0
        
        
        target = torch.cat([target_1.unsqueeze(0), target_2.unsqueeze(0), ref.unsqueeze(0)], dim=0)
        # target = torch.cat([noisy_1.unsqueeze(0), noisy_2.unsqueeze(0), ref.unsqueeze(0)], dim=0)
        noisy = torch.cat([noisy_1.unsqueeze(0), noisy_2.unsqueeze(0), ref.unsqueeze(0)], dim=0)
        prompt = ""            

        input_ids = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        

        return {
            "output_pixel_values": target,
            "conditioning_pixel_values": noisy,
            "caption": prompt,
            "input_ids": input_ids,
            "frame_id": ['frontal_cam', self.frame_list[idx][0].split('_')[-1][:-4], self.frame_list[idx][1].split('_')[-1][:-4]]
        }
        


class InfData(torch.utils.data.Dataset):
    def __init__(self, split, image_prep, tokenizer):
        self.frame_list = []

        self.root_dir = '../nersemble/aligned_test_symlink/'
        self.root_dir1 = '../ava-processed/ava-test-new/'
        
        subjects = os.listdir(self.root_dir)
        for subject in subjects:
            if subject[0] == 'n':
                frames = os.listdir(f'{self.root_dir}{subject}/gp-avatar/')
                # frames = [(subject, frame) for frame in frames]
                frames = [(subject, frames[i*2], frames[i*2 +1]) for i in range(len(frames)//2)]
                self.frame_list.extend(frames)
                
        if self.root_dir1 is not None:
            subjects = os.listdir(self.root_dir1)
            for subject in subjects:
                # if os.path.exists(f'{self.root_dir1}{subject}/gp-avatar/') and len(os.listdir(f'{self.root_dir1}{subject}/gp-avatar/')) >= 1:
                frames = os.listdir(f'{self.root_dir1}{subject}/gp-avatar/')
                if len(frames) != 0:
                    for frontal in ['401168','401875','402040','401031']:
                        frames_front = [f for f in frames if frames[0][3:9] == frontal]
                        # frames = [(subject, frame) for frame in frames]# if frame[3:-4] in ['401168','401875','402040','401031'] and frame != 'dataset.json' and len(glob(f"{self.root_dir1}{subject}/gp-avatar/{frame[:-4]}*")) >= 2]
                        frames_front = [(subject, frames_front[i*2], frames_front[i*2 +1]) for i in range(len(frames_front)//2)]
                        self.frame_list.extend(frames_front)
        
                    
        self.img_resize = Resize((256,256))
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.frame_list)
    
    def __getitem__(self, idx):

        
        if self.frame_list[idx][0][0] == 'n':
            root_dir = self.root_dir
        else:
            root_dir = self.root_dir1
            
        path_1, path_2 = self.frame_list[idx][1], self.frame_list[idx][2]
        
        img1 = Image.open(root_dir + self.frame_list[idx][0] + '/gp-avatar/' + path_1)
        img2 = Image.open(root_dir + self.frame_list[idx][0] + '/gp-avatar/' + path_2)
        
        if self.frame_list[idx][0][0] != 'n':
            img1 = np.array(img1)
            img1 = linear2color_corr(img1)
            img1 = einops.rearrange(torch.tensor(img1), 'h w c -> c h w')
            img2 = np.array(img2)
            img2 = linear2color_corr(img2)
            img2 = einops.rearrange(torch.tensor(img2), 'h w c -> c h w')
        else:
            img1 = pil_to_tensor(img1)
            img2 = pil_to_tensor(img2)
        ref = img1[:,:, :512]
        ref = self.img_resize(ref)
        ref = (ref / 127.5) - 1.0
        
        noisy_1 = img1[:,:,1024:1536]
        noisy_1 = self.img_resize(noisy_1)
        noisy_1 = (noisy_1 / 127.5) - 1.0

        noisy_2 = img2[:,:,1024:1536]
        noisy_2 = self.img_resize(noisy_2)
        noisy_2 = (noisy_2 / 127.5) - 1.0
        
        target_1 = img1[:,:,512:1024]
        target_1 = self.img_resize(target_1)
        target_1 = (target_1 / 127.5) - 1.0
        
        target_2 = img2[:,:,512:1024]
        target_2 = self.img_resize(target_2)
        target_2 = (target_2 / 127.5) - 1.0
        
        
        target = torch.cat([target_1.unsqueeze(0), target_2.unsqueeze(0), ref.unsqueeze(0)], dim=0)
        noisy = torch.cat([noisy_1.unsqueeze(0), noisy_2.unsqueeze(0), ref.unsqueeze(0)], dim=0)
        prompt = ""            

        input_ids = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        

        return {
            "output_pixel_values": target,
            "conditioning_pixel_values": noisy,
            "caption": prompt,
            "input_ids": input_ids,
            "frame_id": [self.frame_list[idx][0], self.frame_list[idx][1].split('_')[-1][:-4], self.frame_list[idx][2].split('_')[-1][:-4]]
        }
        

class CustomData(torch.utils.data.Dataset):
    def __init__(self, split, image_prep, tokenizer):
        self.frame_list = []
        self.root_dir = None
        self.root_dir1 = None
        if split == 'test':
            self.root_dir = '../nersemble/aligned_test_symlink/'
            self.root_dir1 = '../ava-processed/ava-test-new/'
            
        if split == 'train':    
            self.root_dir = '../nersemble/aligned_train_symlink/'
            self.root_dir1 = '../ava-processed/ava-train-new/'
        
        if split == 'pretrain':    
            self.root_dir = '../panohead-samples/'

            frames = os.listdir(self.root_dir + '/crop/')
            frames = [('panohead', frame.split('+')[0]) for frame in frames if 'json' not in frame]
            self.frame_list.extend(frames)

        else:

            subjects = os.listdir(self.root_dir)
            for subject in subjects:
                if subject[0] == 'n':
                     if os.path.exists(f'{self.root_dir}{subject}/gp-avatar/') and len(os.listdir(f'{self.root_dir}{subject}/gp-avatar/')) >= 1:
                        # frames = os.listdir(f'{self.root_dir}{subject}/gp-avatar/')
                        # frames = [(subject, frame) for frame in frames]
                        frames = [(subject, 'cam_222200037')]
                        self.frame_list.extend(frames)

            if self.root_dir1 is not None:
            #     # frames = os.listdir(self.root_dir1 + '/spherehead-imgs/')
            #     # self.frame_list.extend(frames)
                subjects = os.listdir(self.root_dir1)
                for subject in subjects:
                    if os.path.exists(f'{self.root_dir1}{subject}/gp-avatar/') and len(os.listdir(f'{self.root_dir1}{subject}/gp-avatar/')) >= 1:
                        frames = os.listdir(f'{self.root_dir1}{subject}/gp-avatar/')
                        # frames = [(subject, frame) for frame in frames]# if frame[3:-4] in ['401168','401875','402040','401031'] and frame != 'dataset.json' and len(glob(f"{self.root_dir1}{subject}/gp-avatar/{frame[:-4]}*")) >= 2]
                        frames = [(subject, frames[0][:9])]
                        self.frame_list.extend(frames)

        self.img_resize = Resize((256,256))
        self.tokenizer = tokenizer
        

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):

        if self.frame_list[idx][0] == 'panohead':
            gp_avatar = Image.open(f"{self.root_dir}/gp-avatar/{self.frame_list[idx][1]}+view1.png")
            
            # ref = Image.open(f"{self.root_dir}/crop/{self.frame_list[idx][1]}+front.png")
            ref = pil_to_tensor(gp_avatar)[:,:, :512]
            ref = self.img_resize(ref)
            ref = (ref / 127.5) - 1.0
            
            noisy_1_f = Image.open(f"{self.root_dir}/gp-avatar/{self.frame_list[idx][1]}+view1.png")
            noisy_1 = pil_to_tensor(noisy_1_f)[:,:,1024:1536]
            noisy_1 = self.img_resize(noisy_1)
            noisy_1 = (noisy_1 / 127.5) - 1.0

            noisy_2_f = Image.open(f"{self.root_dir}/gp-avatar/{self.frame_list[idx][1]}+view2.png")
            noisy_2 = pil_to_tensor(noisy_2_f)[:,:,1024:1536]
            noisy_2 = self.img_resize(noisy_2)
            noisy_2 = (noisy_2 / 127.5) - 1.0

            prompt = ""

            # target_1 = Image.open(f"{self.root_dir}/cropped_views/{self.frame_list[idx][1]}+view1.png")
            target_1 = pil_to_tensor(noisy_1_f)[:,:,512:1024]
            target_1 = self.img_resize(target_1)
            target_1 = (target_1 / 127.5) - 1.0
            
            # target_2 = Image.open(f"{self.root_dir}/cropped_views/{self.frame_list[idx][1]}+view2.png")
            target_2 = pil_to_tensor(noisy_2_f)[:,:,512:1024]
            target_2 = self.img_resize(target_2)
            target_2 = (target_2 / 127.5) - 1.0

            target = torch.cat([target_1.unsqueeze(0), target_2.unsqueeze(0), ref.unsqueeze(0)], dim=0)
            noisy = torch.cat([noisy_1.unsqueeze(0), noisy_2.unsqueeze(0), ref.unsqueeze(0)], dim=0)
            input_ids = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids

            return {
                "output_pixel_values": target,
                "conditioning_pixel_values": noisy,
                "caption": prompt,
                "input_ids": input_ids,
            }
        else:
            if self.frame_list[idx][0][0] == 'n':
                root_dir = self.root_dir
            else:
                root_dir = self.root_dir1
                
            all_samples = glob(f"{root_dir}{self.frame_list[idx][0]}/gp-avatar/{self.frame_list[idx][1]}*")
            path_1, path_2 = random.sample(all_samples, 2)
            img1 = Image.open(path_1)
            img2 = Image.open(path_2)
            
            if self.frame_list[idx][0][0] != 'n':
                img1 = np.array(img1)
                img1 = linear2color_corr(img1)
                img1 = einops.rearrange(torch.tensor(img1), 'h w c -> c h w')
                img2 = np.array(img2)
                img2 = linear2color_corr(img2)
                img2 = einops.rearrange(torch.tensor(img2), 'h w c -> c h w')
            else:
                img1 = pil_to_tensor(img1)
                img2 = pil_to_tensor(img2)
            ref = img1[:,:, :512]
            ref = self.img_resize(ref)
            ref = (ref / 127.5) - 1.0
            
            noisy_1 = img1[:,:,1024:1536]
            noisy_1 = self.img_resize(noisy_1)
            noisy_1 = (noisy_1 / 127.5) - 1.0

            noisy_2 = img2[:,:,1024:1536]
            noisy_2 = self.img_resize(noisy_2)
            noisy_2 = (noisy_2 / 127.5) - 1.0
            
            target_1 = img1[:,:,512:1024]
            target_1 = self.img_resize(target_1)
            target_1 = (target_1 / 127.5) - 1.0
            
            target_2 = img2[:,:,512:1024]
            target_2 = self.img_resize(target_2)
            target_2 = (target_2 / 127.5) - 1.0
            
            
            target = torch.cat([target_1.unsqueeze(0), target_2.unsqueeze(0), ref.unsqueeze(0)], dim=0)
            noisy = torch.cat([noisy_1.unsqueeze(0), noisy_2.unsqueeze(0), ref.unsqueeze(0)], dim=0)
            prompt = ""            

            input_ids = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            

            return {
                "output_pixel_values": target,
                "conditioning_pixel_values": noisy,
                "caption": prompt,
                "input_ids": input_ids,
            }
            
            

class VideoData(torch.utils.data.Dataset):
    def __init__(self, split, image_prep, tokenizer):
        self.frame_list = {}
        self.root_dir = None
            
        self.root_dir = '/home/emilykim/Desktop/HAR-project/training_data-cropped-rmbg-1/'
        self.kp_dir = '/home/emilykim/Desktop/HAR-project/training_data-cropped-kps/'
        subjects = os.listdir(self.root_dir)
        subjects.sort()
        # if split == 'train':
        #     self.subjects = [subject for subject in self.subjects if 'Subj+026' not in subject and 'Subj+025' not in subject and 'Subj+024' not in subject]
        # else:
        #     self.subjects = [subject for subject in self.subjects if 'Subj+026' in subject or 'Subj+025' in subject or 'Subj+024' in subject]

        self.cumul_frames = 0
        self.num_frames = []
        self.subjects = []
        for subject in subjects:
            frames = os.listdir(self.root_dir + f"/{subject}/real/")
            frames.sort()
            if len(frames) < 2:
                continue
            self.frame_list[self.root_dir + f"/{subject}"] = frames
            self.cumul_frames += len(frames)
            self.num_frames += [len(frames)]
            self.subjects += [subject]

        # self.resize = 
        self.tokenizer = tokenizer
        

    def __len__(self):
        return self.cumul_frames
    
    def __getitem__(self, idx):
        
        subject = None
        switch = True
        while switch:
            rng = 0
            sub_idx, frame_idx = 0, 0
            for i, num in enumerate(self.num_frames):
                if idx < num + rng and idx >= rng:
                    frame_idx = idx - rng
                    sub_idx = i
                    break
                else:
                    rng = num + rng
            
            subject = self.subjects[sub_idx]
            
            frame = self.frame_list[self.root_dir + '/' + subject][frame_idx]
            resize = Resize((384,384))

            next_frame = f'{int(frame[:-4]):04d}.png'
            # if len(self.frame_list[self.root_dir + '/' + subject]) <= 2:
            #     if len(self.frame_list[self.root_dir + '/' + subject]) < idx + 1:
            #         idx += 1
            #     else:
            #         idx -= 1
            #     continue
            ref_frame1 = random.sample(self.frame_list[self.root_dir + '/' + subject][:frame_idx] + self.frame_list[self.root_dir + '/' + subject][frame_idx+1:], 1)[0]

            ref1 = cv2.imread(self.root_dir + '/' + subject + f'/real/{ref_frame1}')
            ref1 = cv2.cvtColor(ref1, cv2.COLOR_BGR2RGB)
            # ref2 = cv2.imread(self.root_dir + '/' + subject + f'/real/{ref_frame2}')
            # ref2 = cv2.cvtColor(ref2, cv2.COLOR_BGR2RGB)
            
            # cur = cv2.imread(self.root_dir + '/' + subject + f'/real/{frame}')
            # cur = cv2.cvtColor(cur, cv2.COLOR_BGR2RGB)
            
            next = cv2.imread(self.root_dir + '/' + subject + f'/real/{next_frame}')
            next = cv2.cvtColor(next, cv2.COLOR_BGR2RGB)
            
            next_keypoints = cv2.imread(self.kp_dir + '/' + subject + f'/{next_frame}')
            next_keypoints = cv2.cvtColor(next_keypoints, cv2.COLOR_BGR2RGB)
            # next = self.align_face_crop(next)
            
            next_noisy = cv2.imread(self.root_dir + '/' + subject + f'/syn/{int(next_frame[:-4]):05d}.png')
            next_noisy = cv2.cvtColor(next_noisy, cv2.COLOR_BGR2RGB)
            
            # output = self.align_face_crop(ref, cur, next, next_noisy)
            # if output is None:
            #     continue
            # switch = False
            # ref, cur, next, next_noisy = output
            
            ref1 = (ref1 / 127.5) - 1.0
            ref1 = torch.tensor(ref1)
            ref1 = einops.rearrange(ref1, 'h w c -> c h w')
            ref1 = resize(ref1)
            
            # ref2 = (ref2 / 127.5) - 1.0
            # ref2 = torch.tensor(ref2)
            # ref2 = einops.rearrange(ref2, 'h w c -> c h w')
            # ref2 = resize(ref2)
            # ref = pad(ref)
            
            # cur = (cur / 127.5) - 1.0
            # cur = torch.tensor(cur)
            # cur = einops.rearrange(cur, 'h w c -> c h w')
            # cur = resize(cur)
            # cur = pad(cur)

            
            next = (next / 127.5) - 1.0
            next = torch.tensor(next)
            next = einops.rearrange(next, 'h w c -> c h w')
            next = resize(next)
            # next = pad(next)
            
            next_keypoints = (next_keypoints / 127.5) - 1.0
            next_keypoints = torch.tensor(next_keypoints)
            next_keypoints = einops.rearrange(next_keypoints, 'h w c -> c h w')
            next_keypoints = resize(next_keypoints)
                    
            
            next_noisy = (next_noisy / 127.5) - 1.0
            next_noisy = torch.tensor(next_noisy)

            next_noisy = einops.rearrange(next_noisy, 'h w c -> c h w')
            next_noisy = resize(next_noisy)
            # next_noisy = pad(next_noisy)
            
            target = torch.cat([ref1.unsqueeze(0), next.unsqueeze(0), next_keypoints.unsqueeze(0)]).to(dtype=torch.bfloat16)
            noisy = torch.cat([ref1.unsqueeze(0), next_noisy.unsqueeze(0), next_keypoints.unsqueeze(0)]).to(dtype=torch.bfloat16)
            
            prompt = "high quality, sharp images"
            
            input_ids = self.tokenizer(
                    prompt, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
            switch = False
            

        return {
            "output_pixel_values": target,
            "conditioning_pixel_values": noisy,
            "caption": prompt,
            "input_ids": input_ids,
        }

def linear2color_corr(img, dim = -1):
    
    if type(img) != np.ndarray:
        img = np.array(img)
    
    img = img / 255
        
    if dim == -1:
        dim = len(img.shape) - 1
    assert img.shape[dim] == 3

    shape = [3 if i == dim else 1 for i in range(len(img.shape))]
    gamma = 1.5254
    black = [4.4 / 255, 3.1 / 255, 4.2 / 255]
    scale = 1.0 / 1.1059
    color_scale = [1.279545, 1.1059, 1.6]

    color_scale = np.array(color_scale, dtype=np.float32)
    color_scale = color_scale.reshape(shape)
    black = np.array(black, dtype=np.float32)
    black = black.reshape(shape)
    img = img * color_scale
    img = (scale / (1 - black)) * (img - black)

    return (np.clip(np.power(img.clip(min=1e-6),(1.0 / gamma)), a_min=0.0, a_max=1.0) * 255).astype(np.uint8)



# class CelebDataset(torch.utils.data.Dataset):
#     def __init__(self, split, image_prep, tokenizer):

#         self.frame_list = os.listdir('../diffusion-to-panohead/for_diffix-ctrlnet/celebA-ctrlnet')
#         self.frame_list = [frame for frame in self.frame_list if '.npy' not in frame]
#         self.img_resize = Resize((256,256))
#         # self.T = build_transform(image_prep)
#         self.tokenizer = tokenizer
        

#     def __len__(self):
#         return len(self.frame_list)

#     def __getitem__(self, idx):
#         camera_transformation = np.append(np.eye(4), np.eye(3))

#         ref = Image.open(f"../ava-processed/CelebAMask-HQ/CelebA-HQ-img/{self.frame_list[idx]}")
#         ref = pil_to_tensor(ref)
#         ref = self.img_resize(ref)
#         ref = (ref / 127.5) - 1.0
#         # ref = einops.rearrange(ref, "c h w -> h w c")
#         # ref = self.T(ref)

#         noisy = Image.open(f"../diffusion-to-panohead/for_diffix-ctrlnet/celebA-ctrlnet/{self.frame_list[idx]}")
#         noisy = pil_to_tensor(noisy)
#         noisy = self.img_resize(noisy)
#         noisy = (noisy / 127.5) - 1.0
#         # noisy = einops.rearrange(noisy, "c h w -> h w c")
        
#         # target = torch.cat([ref.unsqueeze(0), ref.unsqueeze(0)], dim=0)
#         target = ref
#         noisy = torch.cat([noisy.unsqueeze(0), ref.unsqueeze(0)], dim=0)
#         prompt = ""
        

#         input_ids = self.tokenizer(
#             prompt, max_length=self.tokenizer.model_max_length,
#             padding="max_length", truncation=True, return_tensors="pt"
#         ).input_ids
        

#         return {
#             "output_pixel_values": target,
#             "conditioning_pixel_values": noisy,
#             "caption": prompt,
#             "input_ids": input_ids,
#         }

        # return dict(jpg=target.to(torch.bfloat16), txt=prompt, hint=target.to(torch.bfloat16), ref=ref.to(torch.bfloat16), fname=self.frame_list[idx], w=w, w_ref=w_ref, cam_id='', sub_id='CelebA')



    # def __init__(self, dataset_folder, split, image_prep, tokenizer):
    #     """
    #     Itialize the paired dataset object for loading and transforming paired data samples
    #     from specified dataset folders.

    #     This constructor sets up the paths to input and output folders based on the specified 'split',
    #     loads the captions (or prompts) for the input images, and prepares the transformations and
    #     tokenizer to be applied on the data.

    #     Parameters:
    #     - dataset_folder (str): The root folder containing the dataset, expected to include
    #                             sub-folders for different splits (e.g., 'train_A', 'train_B').
    #     - split (str): The dataset split to use ('train' or 'test'), used to select the appropriate
    #                    sub-folders and caption files within the dataset folder.
    #     - image_prep (str): The image preprocessing transformation to apply to each image.
    #     - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
    #     """
    #     super().__init__()
    #     if split == "train":
    #         self.input_folder = os.path.join(dataset_folder, "train_A")
    #         self.output_folder = os.path.join(dataset_folder, "train_B")
    #         captions = os.path.join(dataset_folder, "train_prompts.json")
    #     elif split == "test":
    #         self.input_folder = os.path.join(dataset_folder, "test_A")
    #         self.output_folder = os.path.join(dataset_folder, "test_B")
    #         captions = os.path.join(dataset_folder, "test_prompts.json")
    #     with open(captions, "r") as f:
    #         self.captions = json.load(f)
    #     self.img_names = list(self.captions.keys())
    #     self.T = build_transform(image_prep)
    #     self.tokenizer = tokenizer

    # def __len__(self):
    #     """
    #     Returns:
    #     int: The total number of items in the dataset.
    #     """
    #     return len(self.captions)

    # def __getitem__(self, idx):
    #     """
    #     Retrieves a dataset item given its index. Each item consists of an input image, 
    #     its corresponding output image, the captions associated with the input image, 
    #     and the tokenized form of this caption.

    #     This method performs the necessary preprocessing on both the input and output images, 
    #     including scaling and normalization, as well as tokenizing the caption using a provided tokenizer.

    #     Parameters:
    #     - idx (int): The index of the item to retrieve.

    #     Returns:
    #     dict: A dictionary containing the following key-value pairs:
    #         - "output_pixel_values": a tensor of the preprocessed output image with pixel values 
    #         scaled to [-1, 1].
    #         - "conditioning_pixel_values": a tensor of the preprocessed input image with pixel values 
    #         scaled to [0, 1].
    #         - "caption": the text caption.
    #         - "input_ids": a tensor of the tokenized caption.

    #     Note:
    #     The actual preprocessing steps (scaling and normalization) for images are defined externally 
    #     and passed to this class through the `image_prep` parameter during initialization. The 
    #     tokenization process relies on the `tokenizer` also provided at initialization, which 
    #     should be compatible with the models intended to be used with this dataset.
    #     """
    #     img_name = self.img_names[idx]
    #     input_img = Image.open(os.path.join(self.input_folder, img_name))
    #     output_img = Image.open(os.path.join(self.output_folder, img_name))
    #     caption = self.captions[img_name]

    #     # input images scaled to 0,1
    #     img_t = self.T(input_img)
    #     img_t = F.to_tensor(img_t)
    #     # output images scaled to -1,1
    #     output_t = self.T(output_img)
    #     output_t = F.to_tensor(output_t)
    #     output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

    #     input_ids = self.tokenizer(
    #         caption, max_length=self.tokenizer.model_max_length,
    #         padding="max_length", truncation=True, return_tensors="pt"
    #     ).input_ids

    #     return {
    #         "output_pixel_values": output_t,
    #         "conditioning_pixel_values": img_t,
    #         "caption": caption,
    #         "input_ids": input_ids,
    #     }

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        Itialize the paired dataset object for loading and transforming paired data samples
        from specified dataset folders.

        This constructor sets up the paths to input and output folders based on the specified 'split',
        loads the captions (or prompts) for the input images, and prepares the transformations and
        tokenizer to be applied on the data.

        Parameters:
        - dataset_folder (str): The root folder containing the dataset, expected to include
                                sub-folders for different splits (e.g., 'train_A', 'train_B').
        - split (str): The dataset split to use ('train' or 'test'), used to select the appropriate
                       sub-folders and caption files within the dataset folder.
        - image_prep (str): The image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        if split == "train":
            self.input_folder = os.path.join(dataset_folder, "train_A")
            self.output_folder = os.path.join(dataset_folder, "train_B")
            captions = os.path.join(dataset_folder, "train_prompts.json")
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, "test_A")
            self.output_folder = os.path.join(dataset_folder, "test_B")
            captions = os.path.join(dataset_folder, "test_prompts.json")
        with open(captions, "r") as f:
            self.captions = json.load(f)
        self.img_names = list(self.captions.keys())
        self.T = build_transform(image_prep)
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.captions)

    def __getitem__(self, idx):
        """
        Retrieves a dataset item given its index. Each item consists of an input image, 
        its corresponding output image, the captions associated with the input image, 
        and the tokenized form of this caption.

        This method performs the necessary preprocessing on both the input and output images, 
        including scaling and normalization, as well as tokenizing the caption using a provided tokenizer.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        dict: A dictionary containing the following key-value pairs:
            - "output_pixel_values": a tensor of the preprocessed output image with pixel values 
            scaled to [-1, 1].
            - "conditioning_pixel_values": a tensor of the preprocessed input image with pixel values 
            scaled to [0, 1].
            - "caption": the text caption.
            - "input_ids": a tensor of the tokenized caption.

        Note:
        The actual preprocessing steps (scaling and normalization) for images are defined externally 
        and passed to this class through the `image_prep` parameter during initialization. The 
        tokenization process relies on the `tokenizer` also provided at initialization, which 
        should be compatible with the models intended to be used with this dataset.
        """
        img_name = self.img_names[idx]
        input_img = Image.open(os.path.join(self.input_folder, img_name))
        output_img = Image.open(os.path.join(self.output_folder, img_name))
        caption = self.captions[img_name]

        # input images scaled to 0,1
        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)
        # output images scaled to -1,1
        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "input_ids": input_ids,
        }


class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        A dataset class for loading unpaired data samples from two distinct domains (source and target),
        typically used in unsupervised learning tasks like image-to-image translation.

        The class supports loading images from specified dataset folders, applying predefined image
        preprocessing transformations, and utilizing fixed textual prompts (captions) for each domain,
        tokenized using a provided tokenizer.

        Parameters:
        - dataset_folder (str): Base directory of the dataset containing subdirectories (train_A, train_B, test_A, test_B)
        - split (str): Indicates the dataset split to use. Expected values are 'train' or 'test'.
        - image_prep (str): he image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        if split == "train":
            self.source_folder = os.path.join(dataset_folder, "train_A")
            self.target_folder = os.path.join(dataset_folder, "train_B")
        elif split == "test":
            self.source_folder = os.path.join(dataset_folder, "test_A")
            self.target_folder = os.path.join(dataset_folder, "test_B")
        self.tokenizer = tokenizer
        with open(os.path.join(dataset_folder, "fixed_prompt_a.txt"), "r") as f:
            self.fixed_caption_src = f.read().strip()
            self.input_ids_src = self.tokenizer(
                self.fixed_caption_src, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids

        with open(os.path.join(dataset_folder, "fixed_prompt_b.txt"), "r") as f:
            self.fixed_caption_tgt = f.read().strip()
            self.input_ids_tgt = self.tokenizer(
                self.fixed_caption_tgt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
        # find all images in the source and target folders with all IMG extensions
        self.l_imgs_src = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_src.extend(glob(os.path.join(self.source_folder, ext)))
        self.l_imgs_tgt = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_tgt.extend(glob(os.path.join(self.target_folder, ext)))
        self.T = build_transform(image_prep)

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def __getitem__(self, index):
        """
        Fetches a pair of unaligned images from the source and target domains along with their 
        corresponding tokenized captions.

        For the source domain, if the requested index is within the range of available images,
        the specific image at that index is chosen. If the index exceeds the number of source
        images, a random source image is selected. For the target domain,
        an image is always randomly selected, irrespective of the index, to maintain the 
        unpaired nature of the dataset.

        Both images are preprocessed according to the specified image transformation `T`, and normalized.
        The fixed captions for both domains
        are included along with their tokenized forms.

        Parameters:
        - index (int): The index of the source image to retrieve.

        Returns:
        dict: A dictionary containing processed data for a single training example, with the following keys:
            - "pixel_values_src": The processed source image
            - "pixel_values_tgt": The processed target image
            - "caption_src": The fixed caption of the source domain.
            - "caption_tgt": The fixed caption of the target domain.
            - "input_ids_src": The source domain's fixed caption tokenized.
            - "input_ids_tgt": The target domain's fixed caption tokenized.
        """
        if index < len(self.l_imgs_src):
            img_path_src = self.l_imgs_src[index]
        else:
            img_path_src = random.choice(self.l_imgs_src)
        img_path_tgt = random.choice(self.l_imgs_tgt)
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")
        img_t_src = F.to_tensor(self.T(img_pil_src))
        img_t_tgt = F.to_tensor(self.T(img_pil_tgt))
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])
        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
        }
