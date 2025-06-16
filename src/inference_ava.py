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
from tqdm import tqdm
from mmpretrain import get_model
import argparse
import time

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats

from pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import parse_args_paired_training, PairedDataset, InfData, ITWData

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='lady4-interp', help='the directory to save the output')
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)
    args = parser.parse_args()
    
    if args.model_name == '' != args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')
    
    # initialize the model
    net_pix2pix = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path).cuda()
    net_pix2pix.requires_grad_(False)
    net_pix2pix.set_eval()
    # if args.use_fp16:
    #     model.half()

    # make sure that the input image is a multiple of 8
    # input_image = Image.open(args.input_image).convert('RGB')
    # new_width = input_image.width - input_image.width % 8
    # new_height = input_image.height - input_image.height % 8
    # input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
    # bname = os.path.basename(args.input_image)
    
    dataset_val = ITWData(image_prep=args.test_image_prep, split="test", tokenizer=net_pix2pix.tokenizer)

    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    
    os.makedirs(args.output_dir, exist_ok=True)
    step = 0

    for batch in tqdm(dl_val):
        with torch.no_grad():
            x_src = batch["conditioning_pixel_values"].cuda()
            x_tgt = batch["output_pixel_values"].cuda()
            B, V, C, H, W = x_src.shape
            # forward pass
            start = time.time()
            x_tgt_pred = net_pix2pix(x_src, prompt=["" for _ in range(B)], deterministic=True) # prompt_tokens=batch["input_ids"]
            end = time.time()
            # print((end - start))
            output_img = torch.cat([x_tgt[0][2], x_tgt[0][0], x_tgt[0][1], x_src[0][0], x_src[0][1], x_tgt_pred[0][0], x_tgt_pred[0][1]], dim=2)
            output_pil = transforms.ToPILImage()(output_img.cpu() * 0.5 + 0.5)
            fname = f'{batch["frame_id"][0][0]}+{batch["frame_id"][1][0]}+{batch["frame_id"][2][0]}.png'
            outf = os.path.join(args.output_dir, fname)
            output_pil.save(outf)
            step += 1 