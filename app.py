import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

import torch
import torch.distributed as dist
from PIL import Image
import streamlit as st

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

warnings.filterwarnings('ignore')

EXAMPLE_PROMPT = {
    "t2v-1.3B": {"prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."},
    "t2v-14B": {"prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."},
    "t2i-14B": {"prompt": "一个朴素端庄的美人"},
    "i2v-14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
        "image": "examples/i2v_input.JPG",
    },
}

def main():
    st.title("Wan AI Video and Image Generator")
    
    # User inputs
    task = st.selectbox("Select Task", list(WAN_CONFIGS.keys()), index=1)
    size = st.selectbox("Select Size", list(SIZE_CONFIGS.keys()), index=0)
    frame_num = st.number_input("Frame Number", min_value=1, step=4, value=81)
    ckpt_dir = st.text_input("Checkpoint Directory")
    prompt = st.text_area("Enter Prompt", EXAMPLE_PROMPT.get(task, {}).get("prompt", ""))
    image = st.file_uploader("Upload Image (for I2V tasks)", type=["jpg", "png"]) if "i2v" in task else None
    save_file = st.text_input("Output File Name (Optional)")
    
    if st.button("Generate"):
        args = argparse.Namespace(
            task=task,
            size=size,
            frame_num=frame_num,
            ckpt_dir=ckpt_dir,
            offload_model=False,
            ulysses_size=1,
            ring_size=1,
            t5_fsdp=False,
            t5_cpu=False,
            dit_fsdp=False,
            save_file=save_file or None,
            prompt=prompt,
            use_prompt_extend=False,
            prompt_extend_method="local_qwen",
            prompt_extend_model=None,
            prompt_extend_target_lang="zh",
            base_seed=-1,
            image=image,
            sample_solver='unipc',
            sample_steps=None,
            sample_shift=None,
            sample_guide_scale=5.0,
        )
        generate(args)

def generate(args):
    rank = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.info(f"Generating {args.task} ...")
    cfg = WAN_CONFIGS[args.task]
    
    if "t2v" in args.task or "t2i" in args.task:
        logging.info(f"Using prompt: {args.prompt}")
        wan_t2v = wan.WanT2V(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device, rank=rank)
        video = wan_t2v.generate(
            args.prompt, size=SIZE_CONFIGS[args.size], frame_num=args.frame_num, 
            sample_solver=args.sample_solver, sampling_steps=args.sample_steps, 
            guide_scale=args.sample_guide_scale, seed=args.base_seed
        )
    else:
        img = Image.open(args.image).convert("RGB")
        wan_i2v = wan.WanI2V(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device, rank=rank)
        video = wan_i2v.generate(
            args.prompt, img, max_area=MAX_AREA_CONFIGS[args.size], frame_num=args.frame_num, 
            sample_solver=args.sample_solver, sampling_steps=args.sample_steps, 
            guide_scale=args.sample_guide_scale, seed=args.base_seed
        )
    
    if args.save_file is None:
        args.save_file = f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    
    cache_video(tensor=video[None], save_file=args.save_file, fps=cfg.sample_fps, normalize=True, value_range=(-1, 1))
    
    st.success(f"Generation Complete! Saved as {args.save_file}")
    st.video(args.save_file)

if __name__ == "__main__":
    main()
