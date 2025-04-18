import streamlit as st
import argparse
import os
import sys
import warnings
from PIL import Image

# Import necessary modules
sys.path.insert(0, os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-2]))
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video

warnings.filterwarnings('ignore')

# Global Variables
prompt_expander = None
wan_t2v = None

# Argument Parsing
def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a video from text using Streamlit")
    parser.add_argument("--ckpt_dir", type=str, default="cache", help="Checkpoint directory")
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"], help="Prompt extension method")
    parser.add_argument("--prompt_extend_model", type=str, default=None, help="Prompt extension model")
    return parser.parse_args()

# Prompt Enhancement
def prompt_enc(prompt, tar_lang):
    global prompt_expander
    prompt_output = prompt_expander(prompt, tar_lang=tar_lang.lower())
    return prompt_output.prompt if prompt_output.status else prompt

# Text-to-Video Generation
def t2v_generation(txt2vid_prompt, resolution, sd_steps, guide_scale, shift_scale, seed, n_prompt):
    global wan_t2v
    W, H = map(int, resolution.split("*"))
    video = wan_t2v.generate(txt2vid_prompt, size=(W, H), shift=shift_scale, sampling_steps=sd_steps, guide_scale=guide_scale, n_prompt=n_prompt, seed=seed, offload_model=True)
    cache_video(video[None], save_file="output.mp4", fps=16, normalize=True, value_range=(-1, 1))
    return "output.mp4"

# Streamlit UI
st.title("Text to Video Generation")
st.sidebar.title("Settings")

resolution = st.sidebar.selectbox("Resolution (Width*Height)", ['720*1280', '1280*720', '960*960', '1088*832', '832*1088', '480*832', '832*480', '624*624', '704*544', '544*704'], index=0)

prompt = st.text_input("Prompt", "Describe the video you want to generate")
tar_lang = st.radio("Target Language", ["ZH", "EN"], index=0)
if st.button("Enhance Prompt"):
    prompt = prompt_enc(prompt, tar_lang)
    st.text("Enhanced Prompt: " + prompt)

# Advanced Options
with st.expander("Advanced Settings"):
    sd_steps = st.slider("Diffusion Steps", 1, 1000, 50)
    guide_scale = st.slider("Guide Scale", 0.0, 20.0, 5.0)
    shift_scale = st.slider("Shift Scale", 0.0, 10.0, 5.0)
    seed = st.slider("Seed", -1, 2147483647, -1)
    n_prompt = st.text_input("Negative Prompt", "")

if st.button("Generate Video"):
    video_path = t2v_generation(prompt, resolution, sd_steps, guide_scale, shift_scale, seed, n_prompt)
    st.video(video_path)

# Initialize Prompt Expander
args = _parse_args()
prompt_expander = DashScopePromptExpander(model_name=args.prompt_extend_model) if args.prompt_extend_method == "dashscope" else QwenPromptExpander(model_name=args.prompt_extend_model, device=0)

# Load Model
st.sidebar.text("Initializing Model...")
cfg = WAN_CONFIGS['t2v-14B']
wan_t2v = wan.WanT2V(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0)
