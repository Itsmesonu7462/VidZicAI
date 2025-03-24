import logging
import os
import warnings
import torch
import streamlit as st
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image

warnings.filterwarnings('ignore')

# Streamlit UI
st.title("VIdZicAi Video & Image Generator")

# Model Selection
model_type = st.selectbox("Select Model Type", list(WAN_CONFIGS.keys()))

# Prompt Input
prompt = st.text_area("Enter your prompt", "Describe what you want to generate")

# Resolution Selection
resolution = st.selectbox("Select Resolution", list(SIZE_CONFIGS.keys()))

# Advanced Settings
with st.expander("Advanced Settings"):
    sd_steps = st.slider("Diffusion Steps", 1, 1000, 50)
    guide_scale = st.slider("Guide Scale", 0.0, 20.0, 5.0)
    shift_scale = st.slider("Shift Scale", 0.0, 10.0, 5.0)
    seed = st.slider("Seed", -1, 2147483647, -1)
    n_prompt = st.text_input("Negative Prompt", "")

# Model Loading
@st.cache_resource()
def load_model(model_type):
    cfg = WAN_CONFIGS[model_type]
    checkpoint_dir = "checkpoints/"  # Set the correct directory
    return wan.WanT2V(config=cfg, checkpoint_dir=checkpoint_dir, device_id=0, rank=0)

# Load model
model = load_model(model_type)

# Image Upload (For I2V models)
uploaded_image = None
if "i2v" in model_type:
    uploaded_image = st.file_uploader("Upload Input Image", type=["jpg", "png", "jpeg"])

# Generate Button
if st.button("Generate Output"):
    st.write("Generating... Please wait.")
    if "i2v" in model_type and uploaded_image:
        image = Image.open(uploaded_image)
        output = model.generate(prompt, image, SIZE_CONFIGS[resolution], sd_steps, guide_scale, shift_scale, seed, n_prompt)
        cache_video(output[None], save_file="output.mp4", fps=16, normalize=True, value_range=(-1, 1))
        st.video("output.mp4")
    else:
        output = model.generate(prompt, SIZE_CONFIGS[resolution], sd_steps, guide_scale, shift_scale, seed, n_prompt)
        cache_image(output.squeeze(1)[None], save_file="output.png", normalize=True, value_range=(-1, 1))
        st.image("output.png")