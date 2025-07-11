"""
PyTorch 2.6+ Compatibility Module
Provides utilities for handling the weights_only parameter change in torch.load
"""

import torch
from packaging import version

def get_pytorch_version():
    """Get the current PyTorch version as a parsed version object"""
    return version.parse(torch.__version__)

def is_pytorch_26_or_later():
    """Check if PyTorch version is 2.6 or later"""
    return get_pytorch_version() >= version.parse("2.6.0")

def safe_torch_load(*args, **kwargs):
    """
    Safe wrapper for torch.load that automatically adds weights_only=False for PyTorch 2.6+
    """
    # If PyTorch 2.6 or later, and weights_only not explicitly set, use False
    if is_pytorch_26_or_later():
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
    
    return torch.load(*args, **kwargs)

def check_pytorch_compatibility():
    """Check and report PyTorch compatibility status"""
    version_info = get_pytorch_version()
    if is_pytorch_26_or_later():
        print(f"✅ PyTorch {version_info} detected - using weights_only=False for model loading")
        return True
    else:
        print(f"✅ PyTorch {version_info} detected - no compatibility issues")
        return False

# Check compatibility when module is imported
if __name__ != "__main__":
    try:
        check_pytorch_compatibility()
    except Exception as e:
        print(f"⚠️ PyTorch compatibility check failed: {e}") 