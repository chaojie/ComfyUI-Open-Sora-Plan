

import argparse
import os

import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)

import sys
sys.path.insert(0,f'{comfy_path}/custom_nodes/ComfyUI-Open-Sora-Plan')

import random

import imageio
import torch
from diffusers.schedulers import PNDMScheduler
from huggingface_hub import hf_hub_download
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from datetime import datetime
from typing import List, Union
#import gradio as gr
import numpy as np
#from gradio.components import Textbox, Video, Image
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer

from opensora.models.ae import ae_stride_config, getae, getae_wrapper
from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.models.text_encoder import get_text_enc

from opensora.sample.pipeline_videogen import VideoGenPipeline
from opensora.serve.gradio_utils import block_css, title_markdown, randomize_seed_fn, set_env, examples, DESCRIPTION

class OpenSoraPlanLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path":("STRING",{"default":"/home/admin/Open-Sora-Plan/LanguageBind/Open-Sora-Plan-v1.1.0"}),
                "ae":("STRING",{"default":"CausalVAEModel_4x8x8"}),
                "text_encoder_name":("STRING",{"default":"/home/admin/Open-Sora-Plan/DeepFloyd/t5-v1_1-xxl"}),
                "version":("STRING",{"default":"221x512x512"}),
                "force_images":("BOOLEAN",{"default":False}),
            },
        }

    RETURN_TYPES = ("OpenSoraPlanModel",)
    FUNCTION = "run"
    CATEGORY = "OpenSoraPlan"

    def run(self,model_path,ae,text_encoder_name,version,force_images):
        cuda_device = torch.device('cuda:0')
        cpu_device = torch.device('cpu')

        # Load model:
        transformer_model = LatteT2V.from_pretrained(model_path, subfolder=version, torch_dtype=torch.float16, cache_dir='cache_dir').to(cuda_device)

        vae = getae_wrapper(ae)(model_path, subfolder="vae", cache_dir='cache_dir').to(cpu_device, dtype=torch.float16)
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = 0.25
        vae.vae_scale_factor = ae_stride_config[ae]
        image_size = int(version.split('x')[1])
        #latent_size = (image_size // ae_stride_config[ae][1], image_size // ae_stride_config[ae][2])
        #vae.latent_size = latent_size
        transformer_model.force_images = force_images
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_name, cache_dir="cache_dir")
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, cache_dir="cache_dir",
                                                    torch_dtype=torch.float16).to(cuda_device)

        # set eval mode
        transformer_model.eval()
        vae.eval()
        text_encoder.eval()
        scheduler = PNDMScheduler()
        videogen_pipeline = VideoGenPipeline(vae=vae,
                                            text_encoder=text_encoder,
                                            tokenizer=tokenizer,
                                            scheduler=scheduler,
                                            transformer=transformer_model).to(device=cuda_device)
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
        cuda_device = torch.device('cuda:0')
        cpu_device = torch.device('cpu')
        videogen_pipeline.vae.to(device=cuda_device)
        videogen_pipeline.text_encoder.to(device=cuda_device)
        videogen_pipeline.transformer.to(device=cuda_device)
        videogen_pipeline.to(device=cuda_device)
        #seed = int(randomize_seed_fn(seed, randomize_seed))
        set_env(seed)
        video_length = transformer_model.config.video_length if not force_images else 1
        
        height, width = int(version.split('x')[1]), int(version.split('x')[2])
        num_frames = 1 if video_length == 1 else int(version.split('x')[0])
        print(f'num_frames{num_frames}version{version}')
        videos = videogen_pipeline(prompt,
                                num_frames=num_frames,
                                height=height,
                                width=width,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                enable_temporal_attentions=not force_images,
                                num_images_per_prompt=1,
                                mask_feature=True,
                                ).video

        videogen_pipeline.vae.to(device=cpu_device)
        videogen_pipeline.to(device=cpu_device)
        videogen_pipeline.text_encoder.to(device=cpu_device)
        videogen_pipeline.transformer.to(device=cpu_device)
        torch.cuda.empty_cache()
        print(f'{videos.shape}')
        #videos = videos[0]
        #tmp_save_path = 'tmp.mp4'
        #imageio.mimwrite(tmp_save_path, videos, fps=24, quality=9)  # highest quality is 10, lowest is 0
        #display_model_info = f"Video size: {num_frames}×{height}×{width}, \nSampling Step: {sample_steps}, \nGuidance Scale: {scale}"
        return videos/255.0

class OpenSoraPlanSample:
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

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,model,prompt,num_inference_steps,guidance_scale,seed,force_images):
        videogen_pipeline,transformer_model,version=model
        cuda_device = torch.device('cuda:0')
        cpu_device = torch.device('cpu')
        videogen_pipeline.text_encoder.to(device=cuda_device)
        videogen_pipeline.transformer.to(device=cuda_device)
        videogen_pipeline.to(device=cuda_device)
        videogen_pipeline.vae.to(device=cpu_device)
        #seed = int(randomize_seed_fn(seed, randomize_seed))
        set_env(seed)
        video_length = transformer_model.config.video_length if not force_images else 1
        height, width = int(version.split('x')[1]), int(version.split('x')[2])
        num_frames = 1 if video_length == 1 else int(version.split('x')[0])
        videos = videogen_pipeline(prompt,
                                num_frames=num_frames,
                                height=height,
                                width=width,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                enable_temporal_attentions=not force_images,
                                num_images_per_prompt=1,
                                mask_feature=True,
                                output_type="latents",
                                ).video
        videogen_pipeline.to(device=cpu_device)
        videogen_pipeline.text_encoder.to(device=cpu_device)
        videogen_pipeline.transformer.to(device=cpu_device)
        torch.cuda.empty_cache()

        return ({"samples":videos},)

class OpenSoraPlanDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("OpenSoraPlanModel",),
                "samples": ("LATENT",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "PixArt"

    def run(self,model,samples):
        videogen_pipeline,transformer_model,version=model
        cuda_device = torch.device('cuda:0')
        cpu_device = torch.device('cpu')
        videogen_pipeline.vae.to(device=cuda_device)
        latents=samples["samples"]

        with torch.no_grad():
            video = videogen_pipeline.vae.decode(latents)
            video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().permute(0, 1, 3, 4, 2).contiguous()
            video = video/250.0

        videogen_pipeline.vae.to(device=cpu_device)
        torch.cuda.empty_cache()

        return video

NODE_CLASS_MAPPINGS = {
    "OpenSoraPlanLoader":OpenSoraPlanLoader,
    "OpenSoraPlanRun":OpenSoraPlanRun,
    "OpenSoraPlanSample":OpenSoraPlanSample,
    "OpenSoraPlanDecode":OpenSoraPlanDecode,
}