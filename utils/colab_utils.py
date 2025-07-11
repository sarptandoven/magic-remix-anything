#!/usr/bin/env python3
"""
Magic Hour Remix Anything - Colab Utilities
Common utility functions for Google Colab environment
"""

import gc
import os
import sys
import subprocess
import traceback
import warnings
from pathlib import Path

def suppress_warnings():
    """Suppress common warnings in Colab"""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Suppress specific warnings
    warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
    warnings.filterwarnings("ignore", message=".*numpy.dtype size changed.*")
    warnings.filterwarnings("ignore", message=".*numpy.ufunc size changed.*")

def setup_environment():
    """Setup environment variables and paths for Colab"""
    # Add current directory to Python path
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Set environment variables
    os.environ['PYTHONPATH'] = current_dir
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
    
    # Suppress warnings
    suppress_warnings()

def check_colab_environment():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_system_info():
    """Get system information"""
    info = {
        'is_colab': check_colab_environment(),
        'python_version': sys.version,
        'cuda_available': False,
        'gpu_name': None,
        'gpu_memory': None
    }
    
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except ImportError:
        pass
    
    return info

def print_system_info():
    """Print system information"""
    info = get_system_info()
    
    print("🔍 System Information:")
    print(f"   Environment: {'Google Colab' if info['is_colab'] else 'Local'}")
    print(f"   Python: {info['python_version'].split()[0]}")
    print(f"   CUDA: {'Available' if info['cuda_available'] else 'Not Available'}")
    
    if info['gpu_name']:
        print(f"   GPU: {info['gpu_name']} ({info['gpu_memory']:.2f}GB)")

def get_memory_usage():
    """Get current memory usage"""
    memory_info = {}
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_info['system'] = {
            'used_gb': memory.used / 1024**3,
            'total_gb': memory.total / 1024**3,
            'percent': memory.percent
        }
    except ImportError:
        memory_info['system'] = {'error': 'psutil not available'}
    
    try:
        import torch
        if torch.cuda.is_available():
            memory_info['gpu'] = {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
            }
    except ImportError:
        memory_info['gpu'] = {'error': 'torch not available'}
    
    return memory_info

def print_memory_usage():
    """Print current memory usage"""
    memory_info = get_memory_usage()
    
    print("💾 Memory Usage:")
    
    if 'error' not in memory_info['system']:
        sys_mem = memory_info['system']
        print(f"   System: {sys_mem['used_gb']:.2f}GB / {sys_mem['total_gb']:.2f}GB ({sys_mem['percent']:.1f}%)")
    else:
        print(f"   System: {memory_info['system']['error']}")
    
    if 'error' not in memory_info['gpu']:
        gpu_mem = memory_info['gpu']
        print(f"   GPU: {gpu_mem['allocated_gb']:.2f}GB allocated, {gpu_mem['reserved_gb']:.2f}GB reserved")
    else:
        print(f"   GPU: {memory_info['gpu']['error']}")

def cleanup_memory(verbose=True):
    """Clean up memory"""
    if verbose:
        print("🧹 Cleaning up memory...")
    
    # Python garbage collection
    collected = gc.collect()
    
    # Clear PyTorch cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    
    # Force another garbage collection
    gc.collect()
    
    if verbose:
        print(f"   Collected {collected} objects")

def monitor_memory(threshold=80, cleanup_if_high=True):
    """Monitor memory usage and optionally cleanup if high"""
    memory_info = get_memory_usage()
    
    if 'error' not in memory_info['system']:
        memory_percent = memory_info['system']['percent']
        
        if memory_percent > threshold:
            print(f"⚠️  High memory usage detected: {memory_percent:.1f}%")
            
            if cleanup_if_high:
                cleanup_memory()
                # Check again after cleanup
                memory_info = get_memory_usage()
                if 'error' not in memory_info['system']:
                    new_percent = memory_info['system']['percent']
                    print(f"   Memory after cleanup: {new_percent:.1f}%")
            
            return True  # High memory usage detected
    
    return False  # Memory usage is normal

def check_disk_space(path='.', min_gb=2):
    """Check available disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_gb = free / 1024**3
        
        if free_gb < min_gb:
            print(f"⚠️  Low disk space: {free_gb:.2f}GB available")
            return False
        
        return True
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True  # Assume OK if we can't check

def install_package(package_name, upgrade=False, quiet=True):
    """Install a Python package"""
    cmd = ['pip', 'install']
    
    if upgrade:
        cmd.append('--upgrade')
    
    if quiet:
        cmd.append('-q')
    
    cmd.append(package_name)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print(f"❌ Failed to install {package_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing {package_name}: {e}")
        return False

def check_package_installed(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def ensure_packages(packages, install_missing=True):
    """Ensure required packages are installed"""
    missing_packages = []
    
    for package in packages:
        if not check_package_installed(package):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Missing packages: {missing_packages}")
        
        if install_missing:
            print("🔄 Installing missing packages...")
            for package in missing_packages:
                if install_package(package):
                    print(f"✅ Installed {package}")
                else:
                    print(f"❌ Failed to install {package}")
        
        return False
    
    return True

def create_directories(dirs):
    """Create directories if they don't exist"""
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def safe_import(module_name, fallback=None):
    """Safely import a module with optional fallback"""
    try:
        return __import__(module_name)
    except ImportError as e:
        if fallback:
            print(f"⚠️  Could not import {module_name}, using fallback: {fallback}")
            return fallback
        else:
            print(f"❌ Could not import {module_name}: {e}")
            return None

def run_command(command, description="", capture_output=True, timeout=300):
    """Run a shell command with error handling"""
    if description:
        print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            if description:
                print(f"✅ {description} completed")
            return True, result.stdout
        else:
            print(f"❌ Command failed: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out after {timeout} seconds")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ Command execution failed: {e}")
        return False, str(e)

def download_file(url, filename, description=""):
    """Download a file with progress"""
    if description:
        print(f"📥 {description}...")
    
    try:
        import requests
        from tqdm import tqdm
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def get_colab_pro_status():
    """Check if running on Colab Pro (higher memory/compute)"""
    try:
        memory_info = get_memory_usage()
        if 'error' not in memory_info['system']:
            total_gb = memory_info['system']['total_gb']
            # Colab Pro typically has more RAM
            return total_gb > 15  # Standard Colab usually has ~12GB
    except:
        pass
    
    return False

def optimize_for_colab():
    """Apply Colab-specific optimizations"""
    # Set environment variables for better performance
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    
    # Optimize PyTorch for Colab
    try:
        import torch
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    except ImportError:
        pass
    
    # Setup paths and environment
    setup_environment()
    
    print("⚡ Applied Colab optimizations")

def health_check():
    """Perform a comprehensive health check"""
    print("🏥 System Health Check")
    print("=" * 30)
    
    # System info
    print_system_info()
    print()
    
    # Memory usage
    print_memory_usage()
    print()
    
    # Disk space
    check_disk_space()
    print()
    
    # Memory monitoring
    monitor_memory()
    print()
    
    # Colab Pro status
    is_pro = get_colab_pro_status()
    print(f"🎯 Colab Pro: {'Yes' if is_pro else 'No'}")
    
    print("✅ Health check completed")

# Export main functions
__all__ = [
    'setup_environment',
    'check_colab_environment', 
    'get_system_info',
    'print_system_info',
    'get_memory_usage',
    'print_memory_usage',
    'cleanup_memory',
    'monitor_memory',
    'check_disk_space',
    'install_package',
    'check_package_installed',
    'ensure_packages',
    'safe_import',
    'run_command',
    'download_file',
    'optimize_for_colab',
    'health_check'
] 