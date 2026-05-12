# Magic Hour Remix Setup Instructions

## Prerequisites
- Python 3.10-3.12. Python 3.11 is the safest local default for the current ML dependency stack.
- A CUDA-capable NVIDIA GPU is recommended for local inference. CPU-only and Apple Silicon setups may work for limited tests but will be much slower and may need package changes.
- `git`, `ffmpeg`, and enough disk space for model checkpoints.

## Quick local setup

### 1. Create and activate a virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

If you do not have Python 3.11, use Python 3.10 or 3.12 instead. Avoid Python 3.13 for now because several ML packages may not have compatible wheels.

### 2. Install dependencies

For CUDA Linux machines:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

For macOS or CPU-only development, install the PyTorch build recommended by https://pytorch.org/get-started/locally/ first, then run:
```bash
pip install -r requirements.txt
```

### 3. Install local third-party packages
```bash
pip install -e ./sam
```

GroundingDINO is installed from `requirements.txt` via `groundingdino-py==0.4.0`.

### 4. Download model checkpoints
```bash
bash script/download_ckpt.sh
```

Checkpoints are written to `ckpt/` and `ast_master/pretrained_models/`, both of which are ignored by git.

### 5. Run the app
```bash
python app.py
```

For Colab, prefer the maintained notebook linked from the README: `Magic_Hour_Remix_Anything_Simple.ipynb`.

## Notes
- `script/install.sh` is a convenience script for Colab/Linux-style environments. Review it before running locally because it clones and installs external repositories.
- `colab_setup.py` and `colab_launch.py` are optimized for Google Colab and may install packages or launch Gradio with `share=True`.
- Short, low-resolution videos are recommended for first tests.

## Troubleshooting
- If CUDA packages fail on macOS, reinstall PyTorch using the official macOS command from pytorch.org.
- If imports fail after dependency installation, restart the Python runtime and try again.
- If model files are missing, rerun `bash script/download_ckpt.sh`.
