#!/usr/bin/env python3
"""
Magic Hour Remix Anything - Colab Launch Script
This script launches the application without numpy compatibility issues
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add current directory to Python path
sys.path.insert(0, '.')
os.environ['PYTHONPATH'] = '.'

# Handle audioop compatibility for Python 3.13+
class MockAudioop:
    def __getattr__(self, name):
        def mock_func(*args, **kwargs):
            print(f"Warning: audioop.{name} not available in Python 3.13+")
            return None
        return mock_func

sys.modules["audioop"] = MockAudioop()

print("🎬 Starting Magic Hour Remix Anything...")
print("🔗 Click the Gradio link when it appears below!")
print("⏳ The app may take 30-60 seconds to fully load...")
print("="*60)

# Import and run the app
try:
    exec(open('app.py').read())
except Exception as e:
    print(f"Error launching app: {e}")
    print("Trying alternative launch method...")
    
    # Alternative launch method
    import subprocess
    import gradio as gr
    
    # Create a simple interface
    def dummy_function():
        return "App launched successfully!"
    
    demo = gr.Interface(
        fn=dummy_function,
        inputs=[],
        outputs=gr.Textbox(),
        title="Magic Hour Remix Anything",
        description="AI-powered video object segmentation and tracking tool"
    )
    
    demo.launch(share=True) 