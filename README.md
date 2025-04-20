# VidZicAI: AI-Powered Video & Image Generation

## 📌 Overview
VidZicAI is a Streamlit-based AI application that enables users to generate **videos and images** using advanced text-to-image (T2I), text-to-video (T2V), and image-to-video (I2V) models. The application utilizes **WAN AI models**, making it suitable for creative content generation, artistic designs, and synthetic media production.

## 🚀 Features
- **Text-to-Image (T2I)**: Generate high-quality images from textual prompts.
- **Text-to-Video (T2V)**: Convert text into animated video clips.
- **Image-to-Video (I2V)**: Create videos from still images.
- **Advanced Customization**: Adjust parameters like resolution, diffusion steps, and guidance scale.
- **Streamlit UI**: Interactive and user-friendly interface for easy control.

## 🛠 Setup & Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Itsmesonu7462/VidZicAI.git
cd VidZicAI
```
### download weightage from this particular
- ** url for weightage**- https://huggingface.co/google/umt5-xxl/tree/main
- ** find for ** :- umt5-xxl or umt5-xl(huggingface.io)
-** alternative**:- umt5-small or direct download umt5-small to your pc and change @t5.py to umt5-small
### 2️⃣ Install Dependencies
Ensure you have Python installed (recommended **Python 3.10+**). Then, install required packages:
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
streamlit run app.py --server.port 8501
```
This will start the application. Open your browser and go to:
```
http://localhost:8501
```

## 📁 Project Structure
```
VidZicAI/
│── app.py                # Main Streamlit application
│── requirements.txt      # Required dependencies
│── checkpoints/          # Pretrained model weights (.pth files)
│── wan/                  # Core model and utilities
│── assets/               # Images and example outputs
│── examples/             # Sample inputs for testing
```

## 🧩 Usage Guide
1. **Select a model type** (`T2I`, `T2V`, or `I2V`).
2. **Enter a prompt** describing the image/video you want to generate.
3. **Customize advanced settings** like resolution, diffusion steps, and guidance scale.
4. **Click 'Generate Output'** to create an image or video.
5. **Download the generated media** for use in your projects.

## 🔧 Troubleshooting
### ⚠️ Missing Checkpoints Error
Ensure that the `checkpoints/` folder contains the required `.pth` model files. If missing, download them from the official sources.

### ⚠️ CUDA Initialization Error
If you don't have an NVIDIA GPU, set `device="cpu"` in `app.py` instead of `torch.cuda.current_device()`.

### ⚠️ ModuleNotFoundError (Missing Libraries)
Run:
```bash
pip install -r requirements.txt
```
If errors persist, manually install missing dependencies using:
```bash
pip install <missing-library>
```

## 📜 License
This project is licensed under the MIT License. See `LICENSE` for details.

## 👥 Contributors
- **Itsmesonu7462** – Lead Developer
**Adityaraaj62** – Lead Developer
**vksiwan456** – Lead Developer
- **Collaborators Welcome!** – Feel free to contribute

## ⭐ Support & Contributions
Want to contribute? Feel free to open an **issue** or **pull request** on [GitHub](https://github.com/Itsmesonu7462/VidZicAI).

