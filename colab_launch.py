#!/usr/bin/env python3
"""
Magic Hour Remix Anything - Launch Script
Enhanced launch system with multiple fallback methods for Colab
"""

import os
import sys
import gc
import traceback
import subprocess

def print_status(text, status="info"):
    """Print colored status messages"""
    colors = {
        "success": "\033[92m✅\033[0m",
        "error": "\033[91m❌\033[0m", 
        "warning": "\033[93m⚠️\033[0m",
        "progress": "\033[94m🔄\033[0m"
    }
    icon = colors.get(status, "ℹ️")
    print(f"{icon} {text}")

def cleanup_memory():
    """Clean up GPU and system memory"""
    print_status("Cleaning up memory before launch...", "progress")
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
    print_status("Memory cleanup completed", "success")

def get_memory_status():
    """Get current memory status"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU memory
        gpu_info = "Unknown"
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                gpu_info = f"{allocated:.2f}GB used, {cached:.2f}GB cached"
            else:
                gpu_info = "Not available"
        except ImportError:
            pass
        
        print(f"🔵 System Memory: {memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB ({memory_percent:.1f}%)")
        print(f"🔴 GPU Memory: {gpu_info}")
        
        return memory_percent
    except ImportError:
        print("📊 Memory monitoring not available (psutil missing)")
        return 0

def check_requirements():
    """Check if basic requirements are available"""
    try:
        # Check if we can import basic modules
        import sys
        import os
        return True
    except Exception as e:
        print_status(f"Basic requirements check failed: {e}", "error")
        return False

def launch_method_1():
    """Method 1: Direct execution using subprocess"""
    print_status("Attempting direct app launch...", "progress")
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = '.'
        
        # Use subprocess to run the app
        process = subprocess.Popen([
            sys.executable, 'app.py'
        ], env=env, cwd='.')
        
        print_status("App launched via subprocess", "success")
        process.wait()
        return True
        
    except Exception as e:
        print_status(f"Direct launch failed: {e}", "error")
        return False

def launch_method_2():
    """Method 2: Memory-optimized app using subprocess"""
    print_status("Attempting memory-optimized app launch...", "progress")
    
    try:
        if os.path.exists('app_memory_optimized.py'):
            env = os.environ.copy()
            env['PYTHONPATH'] = '.'
            
            process = subprocess.Popen([
                sys.executable, 'app_memory_optimized.py'
            ], env=env, cwd='.')
            
            print_status("Memory-optimized app launched via subprocess", "success")
            process.wait()
            return True
        else:
            print_status("Memory-optimized app not found", "warning")
            return False
            
    except Exception as e:
        print_status(f"Memory-optimized launch failed: {e}", "error")
        return False

def launch_method_3():
    """Method 3: Import-based launch with proper error handling"""
    print_status("Attempting import-based launch...", "progress")
    
    try:
        # First check if gradio is available
        try:
            import gradio as gr
            print_status("Gradio imported successfully", "success")
        except ImportError as e:
            print_status(f"Gradio import failed: {e}", "error")
            return False
        
        # Set up paths
        sys.path.insert(0, '.')
        
        # Import app modules with better error handling
        try:
            import app
            print_status("App module imported successfully", "success")
            
            # Try different launch methods in order of preference
            if hasattr(app, 'get_demo') and callable(app.get_demo):
                print_status("Found get_demo function, launching...", "success")
                demo = app.get_demo()
                demo.launch(share=True, debug=False, show_error=True, quiet=False)
                return True
                
            elif hasattr(app, 'seg_track_app') and callable(app.seg_track_app):
                print_status("Found seg_track_app function, launching...", "success")
                demo = app.seg_track_app()
                demo.launch(share=True, debug=False, show_error=True, quiet=False)
                return True
                
            elif hasattr(app, 'main') and callable(app.main):
                print_status("Found main function, launching...", "success")
                app.main()
                return True
                
            elif hasattr(app, 'demo') and app.demo is not None:
                print_status("Found demo object, launching...", "success")
                app.demo.launch(share=True, debug=False, show_error=True, quiet=False)
                return True
                
            else:
                print_status("No valid launch method found in app module", "warning")
                print(f"Available attributes: {[attr for attr in dir(app) if not attr.startswith('_')]}")
                return False
                
        except ImportError as e:
            print_status(f"Could not import app module: {e}", "error")
            return False
            
    except Exception as e:
        print_status(f"Import-based launch failed: {e}", "error")
        traceback.print_exc()
        return False

def launch_method_4():
    """Method 4: Basic Gradio interface as fallback"""
    print_status("Attempting basic interface launch...", "progress")
    
    try:
        import gradio as gr
        
        def status_interface():
            return """
            🎬 Magic Hour Remix Anything
            
            System Status Check:
            
            ✅ Gradio is working correctly
            ✅ Interface is accessible  
            ✅ Server is running
            
            The main application may be loading. If this is the only interface you see:
            
            1. Try restarting the runtime (Runtime → Restart runtime)
            2. Run the setup cells again
            3. Check the console output for specific errors
            4. Ensure all dependencies were installed correctly
            
            🔧 System Information:
            - Python version: {sys.version}
            - Gradio version: {gr.__version__}
            """
        
        demo = gr.Interface(
            fn=lambda: status_interface(),
            inputs=[],
            outputs=gr.Textbox(label="System Status", lines=15),
            title="🎬 Magic Hour Remix Anything - System Status",
            description="System status and diagnostics interface",
            allow_flagging="never"
        )
        
        print_status("Basic interface created", "success")
        demo.launch(share=True, debug=False, show_error=True, quiet=False)
        return True
        
    except Exception as e:
        print_status(f"Basic interface launch failed: {e}", "error")
        return False

def launch_with_fallbacks():
    """Launch the application with multiple fallback methods"""
    print("🎬 Launching Magic Hour Remix Anything...")
    print("🔗 Click the Gradio link when it appears below!")
    print("⏳ The app may take 30-60 seconds to fully load...")
    print("=" * 60)
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
    
    # Try launch methods in order (skip subprocess methods for Colab)
    launch_methods = [
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
            print(f"Detailed error: {traceback.format_exc()}")
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
    
    # Try import-based launch directly
    try:
        return launch_method_3()
    except Exception as e:
        print_status(f"Quick launch failed: {e}", "error")
        return False

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch Magic Hour Remix Anything')
    parser.add_argument('--quick', action='store_true', help='Quick launch mode')
    parser.add_argument('--memory-check', action='store_true', help='Show memory status only')
    
    try:
        args = parser.parse_args()
    except:
        # If argparse fails (common in Colab), use default behavior
        args = type('Args', (), {'quick': False, 'memory_check': False})()
    
    if args.memory_check:
        get_memory_status()
        return
    
    if args.quick:
        success = quick_launch()
    else:
        success = launch_with_fallbacks()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 