# 🎬 Magic Hour Remix Anything

**AI-powered video object segmentation and tracking for creative video editing**

Transform your videos with cutting-edge AI technology! Magic Hour Remix Anything allows you to segment, track, and manipulate any objects in your videos using multiple selection methods including text prompts, interactive clicks, drawing, and even audio-based detection.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sarptandoven/magic-remix-anything/blob/main/Magic_Hour_Remix_Anything_Simple.ipynb)
[![GitHub](https://img.shields.io/github/stars/sarptandoven/magic-remix-anything?style=social)](https://github.com/sarptandoven/magic-remix-anything)

---

## 🚀 **Quick Start (Google Colab)**

**The easiest way to get started:**

1. **Click the Colab badge above** or [open this link](https://colab.research.google.com/github/sarptandoven/magic-remix-anything/blob/main/Magic_Hour_Remix_Anything_Simple.ipynb)
2. **Run the 3 setup cells** (takes 5-10 minutes)
3. **Click the Gradio link** when it appears
4. **Upload your video** and start tracking!

*No installation required - everything runs in your browser!*

---

## 🎯 **Features**

### **Object Selection Methods**
- 🔴 **Everything**: Automatically detect all objects in the scene
- 🔵 **Click**: Point-and-click interactive selection
- 🟢 **Text**: Describe objects using natural language
- 🟡 **Stroke**: Draw around objects you want to track
- 🎵 **Audio**: Detect and track sound-making objects

### **AI Models**
- **SAM (Segment Anything)**: State-of-the-art object segmentation
- **AOT (Associating Objects with Transformers)**: Advanced object tracking
- **GroundingDINO**: Text-to-object detection
- **AST (Audio Spectrogram Transformer)**: Audio-based object grounding

### **Technical Features**
- ✅ **Memory-optimized** for Google Colab
- ✅ **Multiple launch methods** with fallback support
- ✅ **Comprehensive error handling**
- ✅ **Progress monitoring** and memory management
- ✅ **Professional UI** with Gradio interface

---

## 📋 **How It Works**

1. **Upload Video**: Support for MP4, AVI, MOV, WebM formats
2. **Select Objects**: Choose from 5 different selection methods
3. **AI Processing**: Advanced models segment and track your objects
4. **Download Results**: Get your processed video with tracked objects

---

## 🛠️ **Local Installation**

If you prefer to run locally:

```bash
# Clone the repository
git clone https://github.com/sarptandoven/magic-remix-anything.git
cd magic-remix-anything

# Run automated setup
python colab_setup.py

# Launch the application
python colab_launch.py
```

### **Requirements**
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

---

## 📖 **Usage Guide**

### **Basic Workflow**

1. **Launch the Application**
   ```python
   python colab_launch.py
   ```

2. **Upload Your Video**
   - Drag and drop or click to upload
   - Wait for the first frame to appear

3. **Select Objects**
   - **Everything Tab**: Click "Segment everything" → Click on detected objects
   - **Click Tab**: Click positive/negative points on your target
   - **Text Tab**: Enter descriptions like "person running" or "red car"
   - **Stroke Tab**: Draw around the object you want to track
   - **Audio Tab**: Upload video with audio → Detect sound sources

4. **Start Tracking**
   - Click "Start Tracking"
   - Wait for AI processing to complete
   - Download your results!

### **Advanced Options**

- **Memory Monitoring**: `python colab_launch.py --memory-check`
- **Quick Launch**: `python colab_launch.py --quick`
- **Memory-Optimized**: `python app_memory_optimized.py`

---

## 🎨 **Use Cases**

- **Video Editing**: Remove or replace objects in videos
- **Content Creation**: Track specific subjects across scenes
- **Sports Analysis**: Follow players or equipment
- **Wildlife Monitoring**: Track animals in nature footage
- **Security**: Monitor specific objects or people
- **Education**: Demonstrate object tracking concepts

---

## 🔧 **Architecture**

Magic Hour Remix Anything integrates multiple state-of-the-art AI models:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gradio UI     │    │   SegTracker     │    │   AI Models     │
│                 │    │                  │    │                 │
│ • Video Upload  │────│ • Coordination   │────│ • SAM           │
│ • Object Select │    │ • Memory Mgmt    │    │ • AOT           │
│ • Progress      │    │ • Error Handle   │    │ • GroundingDINO │
│ • Download      │    │ • Optimization   │    │ • AST           │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Key Components**
- **Frontend**: Gradio web interface with multiple interaction methods
- **Backend**: SegTracker class coordinating all AI models
- **Models**: SAM for segmentation, AOT for tracking, GroundingDINO for text detection
- **Utils**: Memory management, error handling, Colab optimizations

---

## 🚀 **Performance Tips**

- **Use GPU runtime** in Colab for 10x faster processing
- **Shorter videos** (< 30 seconds) process much faster
- **Lower resolution** videos use less memory
- **Fewer objects** selected = faster processing
- **Monitor memory** usage to prevent crashes

---

## 🛠️ **Troubleshooting**

### **Common Issues**

**App Won't Load**
```python
# Try these solutions:
python colab_launch.py --quick  # Quick launch
python colab_setup.py           # Re-run setup
```

**Memory Errors**
```python
# Check memory usage:
python colab_launch.py --memory-check

# Clean up memory:
from utils.colab_utils import cleanup_memory
cleanup_memory()
```

**Import Errors**
```python
# Health check:
from utils.colab_utils import health_check
health_check()
```

---

## 📚 **Documentation**

- **[Setup Instructions](setup_instructions.md)**: Detailed installation guide
- **[Tutorial](tutorial/)**: Step-by-step usage tutorials
- **[API Reference](docs/)**: Technical documentation
- **[Examples](examples/)**: Sample videos and use cases

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
git clone https://github.com/sarptandoven/magic-remix-anything.git
cd magic-remix-anything
pip install -r requirements.txt
python colab_setup.py
```

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **[Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)** by Meta AI
- **[AOT (Associating Objects with Transformers)](https://github.com/yoxu515/aot-benchmark)** 
- **[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)** by IDEA Research
- **[Gradio](https://gradio.app/)** for the amazing web interface framework

---

## 📊 **Stats**

![GitHub stars](https://img.shields.io/github/stars/sarptandoven/magic-remix-anything?style=social)
![GitHub forks](https://img.shields.io/github/forks/sarptandoven/magic-remix-anything?style=social)
![GitHub issues](https://img.shields.io/github/issues/sarptandoven/magic-remix-anything)
![GitHub license](https://img.shields.io/github/license/sarptandoven/magic-remix-anything)

---

**🎬 Start creating amazing videos with AI-powered object tracking today!**

*Made with ❤️ for the creative community* 