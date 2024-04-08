

import argparse
import os

import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)

import sys
sys.path.insert(0,f'{comfy_path}/custom_nodes/ComfyUI-Open-Sora-Plan')

import random

import imageio
import torch
from diffusers import PNDMScheduler
from huggingface_hub import hf_hub_download
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from datetime import datetime
from typing import List, Union
#import gradio as gr
import numpy as np
#from gradio.components import Textbox, Video, Image
from transformers import T5Tokenizer, T5EncoderModel

from opensora.models.ae import ae_stride_config, getae, getae_wrapper
from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.sample.pipeline_videogen import VideoGenPipeline
from opensora.serve.gradio_utils import block_css, title_markdown, randomize_seed_fn, set_env, examples, DESCRIPTION

class OpenSoraPlanLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path":("STRING",{"default":"LanguageBind/Open-Sora-Plan-v1.0.0"}),
                "ae":("STRING",{"default":"CausalVAEModel_4x8x8"}),
                "text_encoder_name":("STRING",{"default":"DeepFloyd/t5-v1_1-xxl"}),
                "version":("STRING",{"default":"65x512x512"}),
                "force_images":("BOOLEAN",{"default":False}),
            },
        }

    RETURN_TYPES = ("OpenSoraPlanModel",)
    FUNCTION = "run"
    CATEGORY = "OpenSoraPlan"

    def run(self,model_path,ae,text_encoder_name,version,force_images):
        device = torch.device('cuda:0')

        # Load model:
        transformer_model = LatteT2V.from_pretrained(model_path, subfolder=version, torch_dtype=torch.float16, cache_dir='cache_dir').to(device)

        vae = getae_wrapper(ae)(model_path, subfolder="vae", cache_dir='cache_dir').to(device, dtype=torch.float16)
        vae.vae.enable_tiling()
        image_size = int(version.split('x')[1])
        latent_size = (image_size // ae_stride_config[ae][1], image_size // ae_stride_config[ae][2])
        vae.latent_size = latent_size
        transformer_model.force_images = force_images
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_name, cache_dir="cache_dir")
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, cache_dir="cache_dir",
                                                    torch_dtype=torch.float16).to(device)

        # set eval mode
        transformer_model.eval()
        vae.eval()
        text_encoder.eval()
        scheduler = PNDMScheduler()
        videogen_pipeline = VideoGenPipeline(vae=vae,
                                            text_encoder=text_encoder,
                                            tokenizer=tokenizer,
                                            scheduler=scheduler,
                                            transformer=transformer_model).to(device=device)
        return ((videogen_pipeline,transformer_model,version),)

class OpenSoraPlanRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("OpenSoraPlanModel",),
                "prompt":("STRING",{"default":""}),
                "num_inference_steps":("INT",{"default":50}),
                "guidance_scale":("FLOAT",{"default":10.0}),
                "seed":("INT",{"default":1234}),
                "force_images":("BOOLEAN",{"default":False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,model,prompt,num_inference_steps,guidance_scale,seed,force_images):
        videogen_pipeline,transformer_model,version=model
        #seed = int(randomize_seed_fn(seed, randomize_seed))
        set_env(seed)
        video_length = transformer_model.config.video_length if not force_images else 1
        height, width = int(version.split('x')[1]), int(version.split('x')[2])
        num_frames = 1 if video_length == 1 else int(version.split('x')[0])
        videos = videogen_pipeline(prompt,
                                video_length=video_length,
                                height=height,
                                width=width,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                enable_temporal_attentions=not force_images,
                                num_images_per_prompt=1,
                                mask_feature=True,
                                ).video

        torch.cuda.empty_cache()
        print(f'{videos.shape}')
        #videos = videos[0]
        #tmp_save_path = 'tmp.mp4'
        #imageio.mimwrite(tmp_save_path, videos, fps=24, quality=9)  # highest quality is 10, lowest is 0
        #display_model_info = f"Video size: {num_frames}×{height}×{width}, \nSampling Step: {sample_steps}, \nGuidance Scale: {scale}"
        return videos/255.0

NODE_CLASS_MAPPINGS = {
    "OpenSoraPlanLoader":OpenSoraPlanLoader,
    "OpenSoraPlanRun":OpenSoraPlanRun,
}