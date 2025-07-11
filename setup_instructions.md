# Magic Hour Remix Setup Instructions

## Prerequisites
- macOS with Apple Silicon
- Python 3.12 (recommended) or Python 3.11 for better compatibility

## Complete Setup Steps

### 1. Create Python 3.12 Environment (Recommended)
```bash
# Install Python 3.12 using pyenv (if you have it)
pyenv install 3.12.0
pyenv local 3.12.0

# Or use Python 3.12 from Homebrew
brew install python@3.12
```

### 2. Set up Virtual Environment
```bash
# Use Python 3.12 for better compatibility
python3.12 -m venv venv
source venv/bin/activate
```

### 3. Install Core Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install Pillow opencv-python matplotlib scikit-image numpy moviepy
pip install gradio==3.39.0 gdown pycocotools timm==0.4.5
pip install transformers supervision addict yapf
```

### 4. Install SAM
```bash
cd sam
pip install -e .
cd ..
```

### 5. Install GroundingDINO (the key missing piece)
```bash
# Install dependencies first
pip install transformers==4.27.4
pip install supervision

# Clone and install GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
```

### 6. Install Pytorch Correlation Extension
```bash
cd Pytorch-Correlation-extension
python setup.py install
cd ..
```

### 7. Download Model Checkpoints
```bash
# The script should now work
bash script/download_ckpt.sh
```

### 8. Run the Application
```bash
python app.py
```

## Alternative: Using Docker
If you continue having issues, consider using Docker:

```bash
# Create a Dockerfile with Python 3.11/3.12 base image
# This ensures a clean environment with proper dependencies
```

## Key Notes
- Python 3.13 has compatibility issues with several dependencies
- Use Python 3.12 or 3.11 for best results
- GroundingDINO requires specific versions of transformers
- All model checkpoints should be in the `ckpt/` directory

## Troubleshooting
- If you get CUDA errors, install CPU-only versions of PyTorch
- If audioop errors persist, the fix is already in app.py
- For M1/M2 Macs, ensure you're using ARM64 compatible packages 