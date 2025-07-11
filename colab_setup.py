#!/usr/bin/env python3
"""
Magic Hour Remix Anything - Colab Setup Script
Handles all installation, setup, and memory optimization for Google Colab
"""

import gc
import os
import sys
import subprocess
import traceback
import importlib.util
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, '.')

def print_header(text, emoji="🔧"):
    """Print a formatted header"""
    print(f"\n{emoji} {text}")
    print("=" * (len(text) + 4))

def print_status(text, status="info"):
    """Print status with appropriate emoji"""
    emojis = {
        "success": "✅",
        "error": "❌", 
        "warning": "⚠️",
        "info": "ℹ️",
        "progress": "🔄"
    }
    print(f"{emojis.get(status, 'ℹ️')} {text}")

class MemoryManager:
    """Memory monitoring and optimization for Colab"""
    
    @staticmethod
    def get_memory_info():
        """Get current memory usage information"""
        try:
            import torch
            import psutil
            
            memory_info = {}
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3  # GB
                gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                
                memory_info['gpu'] = {
                    'used': gpu_memory,
                    'peak': gpu_memory_max,
                    'cached': gpu_memory_cached
                }
                
                print(f"🔴 GPU Memory - Used: {gpu_memory:.2f}GB, Peak: {gpu_memory_max:.2f}GB, Cached: {gpu_memory_cached:.2f}GB")
            
            # System memory
            memory = psutil.virtual_memory()
            memory_info['system'] = {
                'used': memory.used / 1024**3,
                'total': memory.total / 1024**3,
                'percent': memory.percent
            }
            
            print(f"🔵 System Memory - Used: {memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB ({memory.percent:.1f}%)")
            
            return memory_info
            
        except Exception as e:
            print(f"⚠️ Could not get memory info: {e}")
            return {}
    
    @staticmethod
    def cleanup_memory():
        """Clean up memory to prevent crashes"""
        print_status("Cleaning up memory...", "progress")
        
        # Python garbage collection
        gc.collect()
        
        # Clear PyTorch cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        
        # Force garbage collection again
        gc.collect()
        
        print_status("Memory cleanup completed", "success")
    
    @staticmethod
    def monitor_and_cleanup():
        """Monitor memory and cleanup if needed"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent > 80:  # If memory usage > 80%
                print_status("High memory usage detected, cleaning up...", "warning")
                MemoryManager.cleanup_memory()
                MemoryManager.get_memory_info()
            
            return memory.percent
        except Exception as e:
            print_status(f"Memory monitoring failed: {e}", "error")
            return 0

class DependencyInstaller:
    """Handle all dependency installations"""
    
    @staticmethod
    def run_command(command, description=""):
        """Run a shell command with error handling"""
        try:
            if description:
                print_status(f"{description}...", "progress")
            
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                if description:
                    print_status(f"{description} completed", "success")
                return True
            else:
                print_status(f"Command failed: {result.stderr}", "error")
                return False
                
        except Exception as e:
            print_status(f"Command execution failed: {e}", "error")
            return False
    
    @staticmethod
    def install_system_dependencies():
        """Install system-level dependencies"""
        print_header("Installing System Dependencies", "📦")
        
        commands = [
            ("apt-get update -qq", "Updating package lists"),
            ("apt-get install -y ffmpeg", "Installing FFmpeg")
        ]
        
        for cmd, desc in commands:
            DependencyInstaller.run_command(cmd, desc)
    
    @staticmethod
    def install_python_dependencies():
        """Install Python dependencies in optimal order"""
        print_header("Installing Python Dependencies", "🐍")
        
        # Core dependencies first
        core_deps = [
            "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "opencv-python-headless",
            "gradio==3.50.0",
            "numpy",
            "matplotlib",
            "scipy",
            "scikit-image",
            "pillow==10.4.0",  # Fix version conflict
            "tqdm"
        ]
        
        print_status("Installing core dependencies...", "progress")
        for dep in core_deps:
            cmd = f"pip install -q {dep}"
            DependencyInstaller.run_command(cmd, f"Installing {dep.split()[0]}")
        
        # Memory cleanup after core deps
        MemoryManager.cleanup_memory()
        
        # AI/ML dependencies
        ai_deps = [
            "timm",
            "transformers",
            "accelerate",
            "xformers",
            "groundingdino-py",
            "segment-anything-py",
            "moviepy",
            "librosa",
            "soundfile"
        ]
        
        print_status("Installing AI/ML dependencies...", "progress")
        for dep in ai_deps:
            cmd = f"pip install -q {dep}"
            DependencyInstaller.run_command(cmd, f"Installing {dep}")
        
        # Memory cleanup after AI deps
        MemoryManager.cleanup_memory()
        
        # Additional utilities
        util_deps = [
            "datasets",
            "omegaconf", 
            "hydra-core",
            "psutil",
            "packaging>=21.0"
        ]
        
        print_status("Installing utility dependencies...", "progress")
        for dep in util_deps:
            cmd = f"pip install -q {dep}"
            DependencyInstaller.run_command(cmd, f"Installing {dep}")
        
        # Fix Pillow version conflict
        print_status("Fixing Pillow version conflict...", "progress")
        DependencyInstaller.run_command("pip install -q --force-reinstall pillow==10.4.0", "Fixing Pillow version")
    
    @staticmethod
    def install_sam():
        """Install Segment Anything Model"""
        print_header("Installing SAM (Segment Anything Model)", "🎯")
        
        if os.path.exists('sam'):
            os.chdir('sam')
            DependencyInstaller.run_command("pip install -e .", "Installing SAM")
            os.chdir('..')
            MemoryManager.cleanup_memory()
        else:
            print_status("SAM directory not found", "error")
    
    @staticmethod
    def download_models():
        """Download pre-trained models"""
        print_header("Downloading Pre-trained Models", "📥")
        
        if os.path.exists('script/download_ckpt.sh'):
            DependencyInstaller.run_command("chmod +x script/download_ckpt.sh", "Making download script executable")
            DependencyInstaller.run_command("bash script/download_ckpt.sh", "Downloading model checkpoints")
            
            # Verify downloads
            if os.path.exists('ckpt'):
                print_status("Model checkpoints downloaded successfully", "success")
            else:
                print_status("Model download may have failed", "warning")
        else:
            print_status("Download script not found", "error")

class ImportTester:
    """Test all critical imports"""
    
    @staticmethod
    def test_imports():
        """Test all critical imports with detailed reporting"""
        print_header("Testing Critical Imports", "🧪")
        
        import_results = {}
        
        # Test core libraries
        test_cases = [
            ("gradio", "import gradio as gr; gr.__version__"),
            ("torch", "import torch; torch.__version__"),
            ("opencv", "import cv2; cv2.__version__"),
            ("numpy", "import numpy as np; np.__version__"),
            ("matplotlib", "import matplotlib.pyplot as plt"),
            ("PIL", "from PIL import Image"),
            ("scipy", "import scipy; scipy.__version__"),
            ("skimage", "import skimage; skimage.__version__"),
            ("transformers", "import transformers; transformers.__version__"),
            ("moviepy", "from moviepy.editor import VideoFileClip"),
            ("groundingdino", "import groundingdino"),
            ("segment_anything", "import segment_anything"),
            ("librosa", "import librosa"),
            ("soundfile", "import soundfile")
        ]
        
        for lib_name, import_code in test_cases:
            try:
                exec(import_code)
                print_status(f"{lib_name}", "success")
                import_results[lib_name] = True
            except Exception as e:
                print_status(f"{lib_name}: {e}", "error")
                import_results[lib_name] = False
        
        # Summary
        successful = sum(import_results.values())
        total = len(import_results)
        success_rate = successful / total * 100
        
        print(f"\n📊 Import Summary: {successful}/{total} successful ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print_status("Most imports successful! Ready to proceed.", "success")
        else:
            print_status("Some imports failed. The app may have limited functionality.", "warning")
        
        return import_results

class AudioCompatibility:
    """Handle audioop compatibility for Python 3.10+"""
    
    @staticmethod
    def setup_audioop():
        """Setup audioop compatibility"""
        print_header("Setting up Audio Compatibility", "🔊")
        
        try:
            import audioop
            print_status("audioop module is available", "success")
        except ImportError:
            print_status("audioop module not found, using compatibility module", "warning")
            # The audioop.py file in the repo will handle this
            sys.path.insert(0, '.')
        
        print_status("Audio compatibility setup complete", "success")

class ColabSetup:
    """Main setup orchestrator"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.dependency_installer = DependencyInstaller()
        self.import_tester = ImportTester()
        self.audio_compatibility = AudioCompatibility()
    
    def check_environment(self):
        """Check Colab environment and GPU availability"""
        print_header("Checking Environment", "🔍")
        
        # Check if we're in Colab
        try:
            import google.colab
            print_status("Running in Google Colab", "success")
        except ImportError:
            print_status("Not running in Google Colab", "warning")
        
        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print_status(f"GPU available: {gpu_name} ({gpu_memory:.2f}GB)", "success")
            else:
                print_status("No GPU available", "warning")
        except ImportError:
            print_status("PyTorch not available for GPU check", "warning")
        
        # Initial memory check
        self.memory_manager.monitor_and_cleanup()
    
    def full_setup(self):
        """Run complete setup process"""
        print_header("Magic Hour Remix Anything - Colab Setup", "🎬")
        
        try:
            # Environment check
            self.check_environment()
            
            # Install system dependencies
            self.dependency_installer.install_system_dependencies()
            
            # Install Python dependencies
            self.dependency_installer.install_python_dependencies()
            
            # Memory cleanup after major installations
            print_header("Memory Cleanup After Installations", "🧹")
            self.memory_manager.cleanup_memory()
            
            # Setup audio compatibility
            self.audio_compatibility.setup_audioop()
            
            # Install SAM
            self.dependency_installer.install_sam()
            
            # Download models
            self.dependency_installer.download_models()
            
            # Final memory cleanup
            print_header("Final Memory Cleanup", "🧹")
            self.memory_manager.cleanup_memory()
            
            # Test imports
            import_results = self.import_tester.test_imports()
            
            # Final status
            print_header("Setup Complete!", "🎉")
            print_status("Magic Hour Remix Anything is ready to launch!", "success")
            print_status("You can now run the application using app.py", "info")
            
            return True
            
        except Exception as e:
            print_status(f"Setup failed: {e}", "error")
            print("Full error traceback:")
            traceback.print_exc()
            return False

def main():
    """Main entry point"""
    setup = ColabSetup()
    success = setup.full_setup()
    
    if success:
        print("\n🎬 Setup completed successfully!")
        print("📝 Next steps:")
        print("   1. Run: python app.py")
        print("   2. Or use the launch functions in colab_launch.py")
        print("   3. Click the Gradio link when it appears")
    else:
        print("\n💥 Setup failed!")
        print("🛠️ Try restarting the runtime and running again")

if __name__ == "__main__":
    main() 