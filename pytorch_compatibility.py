"""
PyTorch 2.6+ Compatibility Module
Handles the weights_only parameter change in torch.load
"""

import torch
import sys
from packaging import version

def safe_torch_load(*args, **kwargs):
    """
    Safe wrapper for torch.load that handles PyTorch 2.6+ weights_only parameter
    """
    # Get PyTorch version
    pytorch_version = version.parse(torch.__version__)
    
    # If PyTorch 2.6 or later, and weights_only not explicitly set, use False
    if pytorch_version >= version.parse("2.6.0"):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
    
    return torch.load(*args, **kwargs)

def patch_torch_load():
    """
    Monkey patch torch.load to use our safe version
    """
    # Store original torch.load
    if not hasattr(torch, '_original_load'):
        torch._original_load = torch.load
        torch.load = safe_torch_load
        print("✅ PyTorch load compatibility patch applied")

def unpatch_torch_load():
    """
    Restore original torch.load
    """
    if hasattr(torch, '_original_load'):
        torch.load = torch._original_load
        delattr(torch, '_original_load')
        print("🔄 PyTorch load compatibility patch removed")

# Apply patch automatically when module is imported
try:
    patch_torch_load()
except Exception as e:
    print(f"⚠️ Could not apply PyTorch compatibility patch: {e}") 