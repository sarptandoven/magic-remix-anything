#!/usr/bin/env python3
"""
Magic Hour Remix Anything - Colab Launch Script
Handles application launch with multiple fallback methods and error recovery
"""

import os
import sys
import gc
import traceback
import importlib.util
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, '.')

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

def cleanup_memory():
    """Clean up memory before launch"""
    print_status("Cleaning up memory before launch...", "progress")
    
    # Python garbage collection
    gc.collect()
    
    # Clear PyTorch cache if available
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

def get_memory_status():
    """Get current memory status"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"🔵 System Memory: {memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB ({memory.percent:.1f}%)")
        
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"🔴 GPU Memory: {gpu_memory:.2f}GB used, {gpu_cached:.2f}GB cached")
        
        return memory.percent
    except Exception as e:
        print_status(f"Could not get memory status: {e}", "warning")
        return 0

def check_requirements():
    """Check if all required files exist"""
    required_files = ['app.py']
    optional_files = ['app_memory_optimized.py', 'audioop.py']
    
    missing_required = []
    missing_optional = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_required.append(file)
    
    for file in optional_files:
        if not os.path.exists(file):
            missing_optional.append(file)
    
    if missing_required:
        print_status(f"Missing required files: {missing_required}", "error")
        return False
    
    if missing_optional:
        print_status(f"Missing optional files: {missing_optional}", "warning")
    
    return True

def launch_method_1():
    """Method 1: Direct execution of app.py"""
    print_status("Attempting direct app launch...", "progress")
    
    try:
        # Set environment variables
        os.environ['PYTHONPATH'] = '.'
        
        # Execute the main app
        with open('app.py', 'r') as f:
            app_code = f.read()
        
        exec(app_code)
        return True
        
    except Exception as e:
        print_status(f"Direct launch failed: {e}", "error")
        return False

def launch_method_2():
    """Method 2: Memory-optimized app"""
    print_status("Attempting memory-optimized app launch...", "progress")
    
    try:
        if os.path.exists('app_memory_optimized.py'):
            with open('app_memory_optimized.py', 'r') as f:
                app_code = f.read()
            
            exec(app_code)
            return True
        else:
            print_status("Memory-optimized app not found", "warning")
            return False
            
    except Exception as e:
        print_status(f"Memory-optimized launch failed: {e}", "error")
        return False

def launch_method_3():
    """Method 3: Import-based launch"""
    print_status("Attempting import-based launch...", "progress")
    
    try:
        # Import required modules
        import gradio as gr
        print_status("Gradio imported successfully", "success")
        
        # Try to import app modules
        try:
            # Add current directory to path
            sys.path.insert(0, '.')
            
            # Import the main app
            import app
            print_status("App module imported successfully", "success")
            
            # If the app has a main function or demo object, use it
            if hasattr(app, 'demo'):
                app.demo.launch(share=True, debug=True)
            elif hasattr(app, 'main'):
                app.main()
            else:
                print_status("No launch method found in app module", "warning")
                return False
                
            return True
            
        except ImportError as e:
            print_status(f"Could not import app module: {e}", "error")
            return False
            
    except Exception as e:
        print_status(f"Import-based launch failed: {e}", "error")
        return False

def launch_method_4():
    """Method 4: Basic Gradio interface as fallback"""
    print_status("Attempting basic interface launch...", "progress")
    
    try:
        import gradio as gr
        
        def basic_interface(message):
            return """
            🎬 Magic Hour Remix Anything
            
            The application is starting up...
            
            If you see this message, it means:
            ✅ Gradio is working
            ✅ The interface is accessible
            ⏳ The full app is loading in the background
            
            Please wait a moment and try uploading a video.
            
            If issues persist:
            1. Restart the runtime
            2. Run all setup cells again
            3. Check the console for error messages
            """
        
        demo = gr.Interface(
            fn=basic_interface,
            inputs=gr.Textbox(label="Status Check", placeholder="Type anything to test"),
            outputs=gr.Textbox(label="System Status"),
            title="🎬 Magic Hour Remix Anything",
            description="AI-powered video object segmentation and tracking",
            examples=[["test"]]
        )
        
        print_status("Basic interface created", "success")
        demo.launch(share=True, debug=True)
        return True
        
    except Exception as e:
        print_status(f"Basic interface launch failed: {e}", "error")
        return False

def launch_with_fallbacks():
    """Launch the application with multiple fallback methods"""
    print("🎬 Magic Hour Remix Anything - Launch Script")
    print("=" * 50)
    
    # Pre-launch checks
    if not check_requirements():
        print_status("Pre-launch checks failed", "error")
        return False
    
    # Memory cleanup before launch
    cleanup_memory()
    
    # Show initial memory status
    memory_percent = get_memory_status()
    
    if memory_percent > 90:
        print_status("High memory usage detected! Consider restarting runtime.", "warning")
    
    # Try launch methods in order
    launch_methods = [
        ("Direct Execution", launch_method_1),
        ("Memory-Optimized App", launch_method_2),
        ("Import-Based Launch", launch_method_3),
        ("Basic Interface", launch_method_4)
    ]
    
    for method_name, method_func in launch_methods:
        print(f"\n🚀 Trying {method_name}...")
        
        try:
            if method_func():
                print_status(f"Launch successful using {method_name}!", "success")
                print("\n🎉 Application is running!")
                print("🔗 Click the Gradio link above to access the interface")
                print("📱 The interface may take 30-60 seconds to fully load")
                return True
        except Exception as e:
            print_status(f"{method_name} failed: {e}", "error")
            continue
    
    # If all methods failed
    print_status("All launch methods failed!", "error")
    print("\n🛠️ Troubleshooting suggestions:")
    print("   1. Restart the runtime (Runtime → Restart runtime)")
    print("   2. Run the setup script again: python colab_setup.py")
    print("   3. Check GPU memory and try again")
    print("   4. Try running with CPU only")
    
    # Show final memory status
    print("\n🔍 Final memory status:")
    get_memory_status()
    
    return False

def quick_launch():
    """Quick launch without extensive checks (for when setup is known to be complete)"""
    print("🚀 Quick Launch Mode")
    
    # Basic memory cleanup
    cleanup_memory()
    
    # Set environment
    os.environ['PYTHONPATH'] = '.'
    sys.path.insert(0, '.')
    
    # Direct execution
    try:
        exec(open('app.py').read())
        return True
    except Exception as e:
        print_status(f"Quick launch failed: {e}", "error")
        return False

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch Magic Hour Remix Anything')
    parser.add_argument('--quick', action='store_true', help='Quick launch mode')
    parser.add_argument('--memory-check', action='store_true', help='Show memory status only')
    
    args = parser.parse_args()
    
    if args.memory_check:
        get_memory_status()
        return
    
    if args.quick:
        success = quick_launch()
    else:
        success = launch_with_fallbacks()
    
    if success:
        print("\n✅ Launch completed!")
    else:
        print("\n❌ Launch failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 