import streamlit as st
import argparse
import os
import sys
import gc
import warnings
from PIL import Image
#streamlit modified
# Import necessary modules
sys.path.insert(0, os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-2]))
import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video

warnings.filterwarnings('ignore')

# Global Variables
prompt_expander = None
wan_i2v_480P = None
wan_i2v_720P = None

# Argument Parsing
def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a video from an image using Streamlit")
    parser.add_argument("--ckpt_dir_720p", type=str, default=None, help="Checkpoint directory for 720P model")
    parser.add_argument("--ckpt_dir_480p", type=str, default=None, help="Checkpoint directory for 480P model")
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"], help="Prompt extension method")
    parser.add_argument("--prompt_extend_model", type=str, default=None, help="Prompt extension model")
    return parser.parse_args()

# Load Model
def load_model(resolution):
    global wan_i2v_480P, wan_i2v_720P
    if resolution == "720P" and args.ckpt_dir_720p:
        wan_i2v_720P = wan.WanI2V(config=WAN_CONFIGS['i2v-14B'], checkpoint_dir=args.ckpt_dir_720p, device_id=0, rank=0)
    elif resolution == "480P" and args.ckpt_dir_480p:
        wan_i2v_480P = wan.WanI2V(config=WAN_CONFIGS['i2v-14B'], checkpoint_dir=args.ckpt_dir_480p, device_id=0, rank=0)
    else:
        st.warning("Please provide a valid model checkpoint path")

# Prompt Enhancement
def prompt_enc(prompt, image, tar_lang):
    global prompt_expander
    if not image:
        return prompt
    prompt_output = prompt_expander(prompt, image=image, tar_lang=tar_lang.lower())
    return prompt_output.prompt if prompt_output.status else prompt

# Image-to-Video Generation
def i2v_generation(prompt, image, resolution, sd_steps, guide_scale, shift_scale, seed, n_prompt):
    global wan_i2v_480P, wan_i2v_720P
    model = wan_i2v_720P if resolution == '720P' else wan_i2v_480P
    video = model.generate(prompt, image, max_area=MAX_AREA_CONFIGS[resolution], shift=shift_scale, sampling_steps=sd_steps, guide_scale=guide_scale, n_prompt=n_prompt, seed=seed, offload_model=True)
    cache_video(video[None], save_file="output.mp4", fps=16, normalize=True, value_range=(-1, 1))
    return "output.mp4"

# Streamlit UI
st.title("Image to Video Generation")
st.sidebar.title("Settings")

resolution = st.sidebar.selectbox("Resolution", ["------", "720P", "480P"], index=0)
if st.sidebar.button("Load Model"):
    load_model(resolution)

uploaded_image = st.file_uploader("Upload Input Image", type=["jpg", "png", "jpeg"])
prompt = st.text_input("Prompt", "Describe the video you want to generate")
tar_lang = st.radio("Target Language", ["ZH", "EN"], index=0)
if st.button("Enhance Prompt"):
    prompt = prompt_enc(prompt, uploaded_image, tar_lang)
    st.text("Enhanced Prompt: " + prompt)

# Advanced Options
with st.expander("Advanced Settings"):
    sd_steps = st.slider("Diffusion Steps", 1, 1000, 50)
    guide_scale = st.slider("Guide Scale", 0.0, 20.0, 5.0)
    shift_scale = st.slider("Shift Scale", 0.0, 10.0, 5.0)
    seed = st.slider("Seed", -1, 2147483647, -1)
    n_prompt = st.text_input("Negative Prompt", "")

if st.button("Generate Video"):
    if uploaded_image:
        video_path = i2v_generation(prompt, Image.open(uploaded_image), resolution, sd_steps, guide_scale, shift_scale, seed, n_prompt)
        st.video(video_path)
    else:
        st.warning("Please upload an image")

# Initialize Prompt Expander
args = _parse_args()
prompt_expander = DashScopePromptExpander(model_name=args.prompt_extend_model) if args.prompt_extend_method == "dashscope" else QwenPromptExpander(model_name=args.prompt_extend_model, device=0)
