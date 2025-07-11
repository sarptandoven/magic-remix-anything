# Memory-optimized version of app.py
import sys
import warnings
import gc
import torch
warnings.filterwarnings("ignore")

# Memory optimization settings
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Clear GPU memory at startup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✅ GPU memory cleared at startup")

# Fix for Python 3.13 audioop compatibility
class MockAudioop:
    def __getattr__(self, name):
        def mock_func(*args, **kwargs):
            print(f"Warning: audioop.{name} not available in Python 3.13+")
            return None
        return mock_func

sys.modules["audioop"] = MockAudioop()

# GroundingDINO workaround with memory optimization
try:
    import groundingdino
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    print("GroundingDINO successfully imported")
    GROUNDING_DINO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GroundingDINO not available: {e}")
    print("Text-based detection will not work, but other features should function")
    GROUNDING_DINO_AVAILABLE = False
    
    # Create mock classes/functions
    class MockSLConfig:
        def __init__(self, *args, **kwargs):
            pass
    
    def build_model(*args, **kwargs):
        print("Warning: GroundingDINO not available")
        return None
    
    def clean_state_dict(*args, **kwargs):
        return {}
    
    def get_phrases_from_posmap(*args, **kwargs):
        return []
    
    SLConfig = MockSLConfig

from PIL.ImageOps import colorize, scale
import gradio as gr
import importlib
import sys
import os
import pdb
import json
from matplotlib.pyplot import step

from model_args import segtracker_args,sam_args,aot_args
from SegTracker import SegTracker
from tool.transfer_tools import draw_outline, draw_points

import cv2
from PIL import Image
from skimage.morphology.binary import binary_dilation
import argparse
import torch
import time, math
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video, draw_mask
import gc
import numpy as np
import json
from tool.transfer_tools import mask2bbox

try:
    from ast_master.prepare import ASTpredict
    AST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AST audio model not available: {e}")
    AST_AVAILABLE = False
    def ASTpredict():
        return [], []

# MoviePy workaround
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
    print("MoviePy successfully imported")
except ImportError as e:
    print(f"Warning: MoviePy not available: {e}")
    print("Video functionality will be limited")
    MOVIEPY_AVAILABLE = False
    
    class MockVideoFileClip:
        def __init__(self, *args, **kwargs):
            print("Warning: MoviePy not available - video functionality disabled")
            self.audio = None
        
        def iter_frames(self, *args, **kwargs):
            return []
        
        @property
        def fps(self):
            return 30
        
        @property 
        def duration(self):
            return 0
        
        def set_audio(self, audio):
            return self
        
        def write_videofile(self, *args, **kwargs):
            print("Warning: Video output disabled - MoviePy not available")
    
    VideoFileClip = MockVideoFileClip

# Print component availability status
print("\n" + "="*60)
print("COMPONENT AVAILABILITY STATUS:")
print("="*60)
print(f"✓ GroundingDINO:     {'✓ Available' if GROUNDING_DINO_AVAILABLE else '✗ Not Available (text detection disabled)'}")
print(f"✓ AST Audio Model:   {'✓ Available' if AST_AVAILABLE else '✗ Not Available (audio analysis disabled)'}")
print(f"✓ MoviePy:           {'✓ Available' if MOVIEPY_AVAILABLE else '✗ Not Available (video processing limited)'}")
print("="*60)
print("Starting Magic Hour Remix Anything (Memory Optimized)...")
print("="*60 + "\n")

# Memory monitoring function
def monitor_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
    else:
        print("GPU not available")

# Clean memory function
def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ Memory cleaned")

def clean():
    clean_memory()
    return None, None, None, None, None, None, [[], []]

def audio_to_text(input_video, label_num, threshold):
    if not MOVIEPY_AVAILABLE:
        return "MoviePy not available", {}
    
    try:
        video = VideoFileClip(input_video)      
        audio = video.audio      
        video_without_audio = video.set_audio(None)      
        video_without_audio.write_videofile("video_without_audio.mp4")        
        audio.write_audiofile("audio.flac", codec="flac") 
        top_labels,top_labels_probs = ASTpredict()
        top_labels_and_probs = "{"  
        predicted_texts = ""
        for k in range(10):
            if(k<label_num and top_labels_probs[k]>threshold):
                    top_labels_and_probs += f"\"{top_labels[k]}\": {top_labels_probs[k]:.4f},"
                    predicted_texts +=top_labels[k]+ ' '
            k+=1
        top_labels_and_probs = top_labels_and_probs[:-1]
        top_labels_and_probs += "}"
        top_labels_and_probs_dic = json.loads(top_labels_and_probs)
        print(top_labels_and_probs_dic) 
        return predicted_texts, top_labels_and_probs_dic
    except Exception as e:
        print(f"Error in audio processing: {e}")
        return "Error processing audio", {}

def get_click_prompt(click_stack, point):
    click_stack[0].append(point["coord"])
    click_stack[1].append(point["mode"])
    
    prompt = {
        "points_coord":click_stack[0],
        "points_mode":click_stack[1],
        "multimask":"True",
    }
    return prompt

def get_meta_from_video(input_video):
    if input_video is None:
        return None, None, None, ""

    print("get meta information of input video")
    cap = cv2.VideoCapture(input_video)
    
    _, first_frame = cap.read()
    cap.release()

    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    clean_memory()
    return first_frame, first_frame, first_frame, ""

def get_meta_from_img_seq(input_img_seq):
    if input_img_seq is None:
        return None, None, None, ""

    print("get meta information of img seq")
    # Create dir
    file_name = input_img_seq.name.split('/')[-1].split('.')[0]
    file_path = f'./assets/{file_name}'
    if os.path.isdir(file_path):
        os.system(f'rm -r {file_path}')
    os.makedirs(file_path)
    # Unzip file
    os.system(f'unzip {input_img_seq.name} -d ./assets ')
    
    imgs_path = sorted([os.path.join(file_path, img_name) for img_name in os.listdir(file_path)])
    first_frame = imgs_path[0]
    first_frame = cv2.imread(first_frame)
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    clean_memory()
    return first_frame, first_frame, first_frame, ""

def SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
    with torch.cuda.amp.autocast():
        Seg_Tracker.add_reference(origin_frame, predicted_mask)
        Seg_Tracker.first_frame_mask = predicted_mask
    clean_memory()
    return Seg_Tracker

def init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame):
    if origin_frame is None:
        return None, None, [[], []], ""
    
    print("init_SegTracker")
    sam_gap = int(sam_gap)
    max_obj_num = int(max_obj_num)
    points_per_side = int(points_per_side)
    long_term_mem = int(long_term_mem)
    max_len_long_term = int(max_len_long_term)
    
    sam_args = sam_args
    sam_args['generator_args']['points_per_side'] = points_per_side
    sam_args['generator_args']['pred_iou_thresh'] = 0.86
    sam_args['generator_args']['stability_score_thresh'] = 0.92
    sam_args['generator_args']['max_detections'] = max_obj_num
    sam_args['generator_args']['crop_n_layers'] = 1
    sam_args['generator_args']['crop_n_points_downscale_factor'] = 2
    sam_args['generator_args']['min_mask_region_area'] = 100
    sam_args['generator_args']['output_mode'] = "binary_mask"
    
    segtracker_args = segtracker_args
    segtracker_args['sam_gap'] = sam_gap
    segtracker_args['min_area'] = 200
    segtracker_args['max_obj_num'] = max_obj_num
    segtracker_args['min_new_obj_iou'] = 0.8
    
    aot_args = aot_args
    aot_args['model'] = aot_model
    aot_args['model_path'] = aot_model2ckpt[aot_model]
    aot_args['long_term_mem_gap'] = long_term_mem
    aot_args['max_len_long_term'] = max_len_long_term
    
    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    clean_memory()
    return Seg_Tracker, origin_frame, [[], []], ""

def init_SegTracker_Stroke(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame):
    if origin_frame is None:
        return None, None, [[], []], None
    
    print("init_SegTracker_Stroke")
    sam_gap = int(sam_gap)
    max_obj_num = int(max_obj_num)
    points_per_side = int(points_per_side)
    long_term_mem = int(long_term_mem)
    max_len_long_term = int(max_len_long_term)
    
    sam_args = sam_args
    sam_args['generator_args']['points_per_side'] = points_per_side
    sam_args['generator_args']['pred_iou_thresh'] = 0.86
    sam_args['generator_args']['stability_score_thresh'] = 0.92
    sam_args['generator_args']['max_detections'] = max_obj_num
    sam_args['generator_args']['crop_n_layers'] = 1
    sam_args['generator_args']['crop_n_points_downscale_factor'] = 2
    sam_args['generator_args']['min_mask_region_area'] = 100
    sam_args['generator_args']['output_mode'] = "binary_mask"
    
    segtracker_args = segtracker_args
    segtracker_args['sam_gap'] = sam_gap
    segtracker_args['min_area'] = 200
    segtracker_args['max_obj_num'] = max_obj_num
    segtracker_args['min_new_obj_iou'] = 0.8
    
    aot_args = aot_args
    aot_args['model'] = aot_model
    aot_args['model_path'] = aot_model2ckpt[aot_model]
    aot_args['long_term_mem_gap'] = long_term_mem
    aot_args['max_len_long_term'] = max_len_long_term
    
    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    clean_memory()
    return Seg_Tracker, origin_frame, [[], []], None

def undo_click_stack_and_refine_seg(Seg_Tracker, origin_frame, click_stack, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side):
    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)
    
    print("Undo")
    if len(click_stack[0]) > 0:
        click_stack[0] = click_stack[0][:-1]
        click_stack[1] = click_stack[1][:-1]
    
    if len(click_stack[0]) > 0:
        click_prompt = get_click_prompt(click_stack, {"coord": [0, 0], "mode": 1})
        masked_frame = seg_acc_click(Seg_Tracker, click_prompt, origin_frame)
    else:
        masked_frame = origin_frame.copy()
    
    clean_memory()
    return Seg_Tracker, masked_frame, click_stack

def roll_back_undo_click_stack_and_refine_seg(Seg_Tracker, origin_frame, click_stack, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side,input_video, input_img_seq, frame_num, refine_idx):
    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)
    
    print("Undo")
    if len(click_stack[0]) > 0:
        click_stack[0] = click_stack[0][:-1]
        click_stack[1] = click_stack[1][:-1]
    
    chosen_frame_show, curr_mask, ori_frame = res_by_num(input_video, input_img_seq, frame_num)
    Seg_Tracker.curr_idx = refine_idx
    
    if len(click_stack[0]) > 0:
        prompt = get_click_prompt(click_stack, {"coord": [0, 0], "mode": 1})
        predicted_mask, masked_frame = Seg_Tracker.seg_acc_click( 
                                                          origin_frame=origin_frame, 
                                                          coords=np.array(prompt["points_coord"]),
                                                          modes=np.array(prompt["points_mode"]),
                                                          multimask=prompt["multimask"],
                                                        )
        curr_mask[curr_mask == refine_idx]  = 0
        curr_mask[predicted_mask != 0]  = refine_idx
        predicted_mask=curr_mask
        Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)
    else:
        masked_frame = origin_frame.copy()
    
    clean_memory()
    return Seg_Tracker, masked_frame, click_stack

def seg_acc_click(Seg_Tracker, prompt, origin_frame):
    # seg acc to click
    predicted_mask, masked_frame = Seg_Tracker.seg_acc_click( 
                                                      origin_frame=origin_frame, 
                                                      coords=np.array(prompt["points_coord"]),
                                                      modes=np.array(prompt["points_mode"]),
                                                      multimask=prompt["multimask"],
                                                    )
    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)
    clean_memory()
    return masked_frame

def sam_click(Seg_Tracker, origin_frame, point_mode, click_stack, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, evt:gr.SelectData):
    print("Click")
    
    if point_mode == "Positive":
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 1}
    else:
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 0}

    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)

    # get click prompts for sam to predict mask
    click_prompt = get_click_prompt(click_stack, point)

    # Refine acc to prompt
    masked_frame = seg_acc_click(Seg_Tracker, click_prompt, origin_frame)
    clean_memory()
    return Seg_Tracker, masked_frame, click_stack

def roll_back_sam_click(Seg_Tracker, origin_frame, point_mode, click_stack, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, input_video, input_img_seq, frame_num, refine_idx, evt:gr.SelectData):
    print("Click")
    
    if point_mode == "Positive":
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 1}
    else:
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 0}

    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)

    # get click prompts for sam to predict mask
    prompt = get_click_prompt(click_stack, point)

    chosen_frame_show, curr_mask, ori_frame = res_by_num(input_video, input_img_seq, frame_num)
    Seg_Tracker.curr_idx = refine_idx

    predicted_mask, masked_frame = Seg_Tracker.seg_acc_click( 
                                                      origin_frame=origin_frame, 
                                                      coords=np.array(prompt["points_coord"]),
                                                      modes=np.array(prompt["points_mode"]),
                                                      multimask=prompt["multimask"],
                                                    )
    curr_mask[curr_mask == refine_idx]  = 0
    curr_mask[predicted_mask != 0]  = refine_idx
    predicted_mask=curr_mask
    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)
    clean_memory()
    return Seg_Tracker, masked_frame, click_stack

def sam_stroke(Seg_Tracker, origin_frame, drawing_board, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side):
    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)

    print("Stroke")
    predicted_mask, masked_frame = Seg_Tracker.seg_acc_stroke(origin_frame, drawing_board)
    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)
    clean_memory()
    return Seg_Tracker, masked_frame, drawing_board

def gd_detect(Seg_Tracker, origin_frame, grounding_caption, box_threshold, text_threshold, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side):
    if Seg_Tracker is None:
        Seg_Tracker, _ , _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)

    print("Detect")
    predicted_mask, annotated_frame= Seg_Tracker.detect_and_seg(origin_frame, grounding_caption, box_threshold, text_threshold)
    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)
    masked_frame = draw_mask(annotated_frame, predicted_mask)
    clean_memory()
    return Seg_Tracker, masked_frame, origin_frame

def segment_everything(Seg_Tracker, aot_model, long_term_mem, max_len_long_term, origin_frame, sam_gap, max_obj_num, points_per_side):
    if Seg_Tracker is None:
        Seg_Tracker, _ , _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)

    print("Everything")
    frame_idx = 0

    with torch.cuda.amp.autocast():
        pred_mask = Seg_Tracker.seg(origin_frame)
        torch.cuda.empty_cache()
        gc.collect()
        Seg_Tracker.add_reference(origin_frame, pred_mask, frame_idx)
        Seg_Tracker.first_frame_mask = pred_mask

    masked_frame = draw_mask(origin_frame.copy(), pred_mask)
    clean_memory()
    return Seg_Tracker, masked_frame

def add_new_object(Seg_Tracker):
    if Seg_Tracker is None:
        return Seg_Tracker, [[], []]
    
    Seg_Tracker.curr_idx += 1
    clean_memory()
    return Seg_Tracker, [[], []]

def tracking_objects(Seg_Tracker, input_video, input_img_seq, fps, frame_num=0):
    if Seg_Tracker is None:
        return None, None
    
    print("Tracking objects...")
    output_video, output_mask = tracking_objects_in_video(Seg_Tracker, input_video, input_img_seq, fps, frame_num)
    clean_memory()
    return output_video, output_mask

def res_by_num(input_video, input_img_seq, frame_num):
    if input_video is None and input_img_seq is None:
        return None, None, None
    
    if input_video is not None:
        cap = cv2.VideoCapture(input_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = None
    else:
        file_name = input_img_seq.name.split('/')[-1].split('.')[0]
        file_path = f'./assets/{file_name}'
        imgs_path = sorted([os.path.join(file_path, img_name) for img_name in os.listdir(file_path)])
        if frame_num < len(imgs_path):
            frame = cv2.imread(imgs_path[frame_num])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = None
    
    if frame is None:
        return None, None, None
    
    # Create a simple mask for visualization
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[frame.shape[0]//4:3*frame.shape[0]//4, frame.shape[1]//4:3*frame.shape[1]//4] = 1
    
    clean_memory()
    return frame, mask, frame

def show_res_by_slider(input_video, input_img_seq, frame_per):
    if input_video is None and input_img_seq is None:
        return None, None
    
    if input_video is not None:
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        frame_num = int(frame_per * total_frames / 100)
        frame_num = max(0, min(frame_num, total_frames - 1))
        
        cap = cv2.VideoCapture(input_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = None
    else:
        file_name = input_img_seq.name.split('/')[-1].split('.')[0]
        file_path = f'./assets/{file_name}'
        imgs_path = sorted([os.path.join(file_path, img_name) for img_name in os.listdir(file_path)])
        
        frame_num = int(frame_per * len(imgs_path) / 100)
        frame_num = max(0, min(frame_num, len(imgs_path) - 1))
        
        if frame_num < len(imgs_path):
            frame = cv2.imread(imgs_path[frame_num])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = None
    
    if frame is None:
        return None, None
    
    # Create a simple mask for visualization
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[frame.shape[0]//4:3*frame.shape[0]//4, frame.shape[1]//4:3*frame.shape[1]//4] = 1
    
    clean_memory()
    return frame, mask

def choose_obj_to_refine(input_video, input_img_seq, Seg_Tracker, frame_num, evt:gr.SelectData):
    if Seg_Tracker is None:
        return None, 0
    
    print("Choose object to refine")
    refine_idx = evt.index[0]
    clean_memory()
    return None, refine_idx

def show_chosen_idx_to_refine(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, input_video, input_img_seq, Seg_Tracker, frame_num, idx):
    if Seg_Tracker is None:
        return None, None, None
    
    print("Show chosen idx to refine")
    chosen_frame_show, curr_mask, ori_frame = res_by_num(input_video, input_img_seq, frame_num)
    Seg_Tracker.curr_idx = idx
    
    if curr_mask is not None:
        curr_mask[curr_mask != idx] = 0
        curr_mask[curr_mask == idx] = idx
        masked_frame = draw_mask(chosen_frame_show.copy(), curr_mask)
    else:
        masked_frame = chosen_frame_show.copy()
    
    clean_memory()
    return Seg_Tracker, masked_frame, chosen_frame_show

def seg_track_app():
    ##########################################################
    ######################  Front-end ########################
    ##########################################################
    
    # Memory monitoring at startup
    monitor_memory()
    
    with gr.Blocks(css="#component-0 {height: 100vh}") as demo:
        gr.Markdown(
            """
            # 🎬 Magic Hour Remix Anything
            **AI-powered video object segmentation and tracking tool**
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎥 Input")
                input_video = gr.Video(label="Input Video", elem_id="input_video")
                input_img_seq = gr.File(label="Input Image Sequence (ZIP)", elem_id="input_img_seq")
                
                with gr.Row():
                    extract_button = gr.Button("Extract First Frame", variant="primary")
                
                gr.Markdown("### ⚙️ Parameters")
                with gr.Row():
                    aot_model = gr.Dropdown(
                        choices=["R50_DeAOTL", "R50_DeAOTB", "R50_DeAOTS", "R50_DeAOTT"],
                        value="R50_DeAOTL",
                        label="AOT Model",
                        elem_id="aot_model"
                    )
                    long_term_mem = gr.Slider(1, 9999, value=9999, step=1, label="Long Term Memory Gap", elem_id="long_term_mem")
                
                with gr.Row():
                    max_len_long_term = gr.Slider(1, 9999, value=9999, step=1, label="Max Length Long Term", elem_id="max_len_long_term")
                    sam_gap = gr.Slider(1, 9999, value=9999, step=1, label="SAM Gap", elem_id="sam_gap")
                
                with gr.Row():
                    max_obj_num = gr.Slider(1, 10, value=10, step=1, label="Max Object Number", elem_id="max_obj_num")
                    points_per_side = gr.Slider(1, 100, value=32, step=1, label="Points Per Side", elem_id="points_per_side")
                
                gr.Markdown("### 🎵 Audio Parameters")
                with gr.Row():
                    label_num = gr.Slider(1, 10, value=5, step=1, label="Label Number", elem_id="label_num")
                    threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Threshold", elem_id="threshold")
                
                gr.Markdown("### 📊 Memory Status")
                memory_status = gr.Textbox(label="Memory Status", value="Ready", interactive=False)
                
                # Memory monitoring button
                monitor_btn = gr.Button("Monitor Memory", variant="secondary")
                monitor_btn.click(fn=lambda: f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB" if torch.cuda.is_available() else "GPU not available", outputs=memory_status)
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎯 Object Selection")
                
                tab_everything = gr.Tab("Everything")
                tab_click = gr.Tab("Click")
                tab_stroke = gr.Tab("Stroke")
                tab_text = gr.Tab("Text")
                tab_audio_grounding = gr.Tab("Audio")
                
                with tab_everything:
                    seg_every_first_frame = gr.Button("Segment Everything for First Frame", variant="primary")
                
                with tab_click:
                    point_mode = gr.Radio(["Positive", "Negative"], value="Positive", label="Point Mode", elem_id="point_mode")
                    click_stack = gr.State([[], []])
                
                with tab_stroke:
                    drawing_board = gr.Image(label="Drawing Board", elem_id="drawing_board")
                    seg_acc_stroke = gr.Button("Segment", variant="primary")
                
                with tab_text:
                    grounding_caption = gr.Textbox(label="Grounding Caption", placeholder="Enter object description...", elem_id="grounding_caption")
                    with gr.Row():
                        box_threshold = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="Box Threshold", elem_id="box_threshold")
                        text_threshold = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Text Threshold", elem_id="text_threshold")
                    detect_button = gr.Button("Detect", variant="primary")
                
                with tab_audio_grounding:
                    audio_to_text_button = gr.Button("Detect the label of the sound-making object", variant="primary")
                    predicted_texts = gr.Textbox(label="Predicted Texts", elem_id="predicted_texts")
                    top_labels_and_probs_dic = gr.JSON(label="Top Labels and Probabilities", elem_id="top_labels_and_probs_dic")
                    audio_grounding_button = gr.Button("Ground the sound-making object", variant="primary")
                
                gr.Markdown("### 🎬 Tracking")
                with gr.Row():
                    fps = gr.Slider(1, 60, value=30, step=1, label="FPS", elem_id="fps")
                    track_for_video = gr.Button("Start Tracking", variant="primary")
                
                gr.Markdown("### 📈 Results")
                output_video = gr.Video(label="Output Video", elem_id="output_video")
                output_mask = gr.File(label="Output Mask", elem_id="output_mask")
                
                gr.Markdown("### 🔧 Refinement")
                with gr.Row():
                    frame_num = gr.Slider(0, 100, value=0, step=1, label="Frame Number", elem_id="frame_num")
                    refine_idx = gr.Slider(0, 10, value=0, step=1, label="Refine Index", elem_id="refine_idx")
                
                output_res = gr.Image(label="Output Result", elem_id="output_res")
                
                # Memory cleanup button
                cleanup_btn = gr.Button("Clean Memory", variant="secondary")
                cleanup_btn.click(fn=clean_memory, outputs=None)
        
        # State variables
        Seg_Tracker = gr.State(None)
        input_first_frame = gr.Image(label="First Frame", elem_id="input_first_frame")
        origin_frame = gr.State(None)
        
        # Input component
        tab_video_input = gr.Tab("Video Input")
        tab_img_seq_input = gr.Tab("Image Sequence Input")
        
        tab_video_input.select(fn=clean, inputs=[], outputs=[input_video, input_img_seq, Seg_Tracker, input_first_frame, origin_frame, drawing_board, click_stack])
        tab_img_seq_input.select(fn=clean, inputs=[], outputs=[input_video, input_img_seq, Seg_Tracker, input_first_frame, origin_frame, drawing_board, click_stack])
        
        extract_button.click(fn=get_meta_from_img_seq, inputs=[input_img_seq], outputs=[input_first_frame, origin_frame, drawing_board, grounding_caption])
        
        # Interactive component
        tab_everything.select(fn=init_SegTracker, inputs=[aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame], outputs=[Seg_Tracker, input_first_frame, click_stack, grounding_caption], queue=False)
        tab_click.select(fn=init_SegTracker, inputs=[aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame], outputs=[Seg_Tracker, input_first_frame, click_stack, grounding_caption], queue=False)
        tab_stroke.select(fn=init_SegTracker_Stroke, inputs=[aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame], outputs=[Seg_Tracker, input_first_frame, click_stack, drawing_board], queue=False)
        tab_text.select(fn=init_SegTracker, inputs=[aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame], outputs=[Seg_Tracker, input_first_frame, click_stack, grounding_caption], queue=False)
        tab_audio_grounding.select(fn=init_SegTracker, inputs=[aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame], outputs=[Seg_Tracker, input_first_frame, click_stack, grounding_caption], queue=False)
        
        audio_to_text_button.click(fn=audio_to_text, inputs=[input_video, label_num, threshold], outputs=[predicted_texts, top_labels_and_probs_dic])
        audio_grounding_button.click(fn=gd_detect, inputs=[Seg_Tracker, origin_frame, predicted_texts, box_threshold, text_threshold, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side], outputs=[Seg_Tracker, input_first_frame])
        
        seg_every_first_frame.click(fn=segment_everything, inputs=[Seg_Tracker, aot_model, long_term_mem, max_len_long_term, origin_frame, sam_gap, max_obj_num, points_per_side], outputs=[Seg_Tracker, input_first_frame])
        
        input_first_frame.select(fn=sam_click, inputs=[Seg_Tracker, origin_frame, point_mode, click_stack, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side], outputs=[Seg_Tracker, input_first_frame, click_stack])
        
        seg_acc_stroke.click(fn=sam_stroke, inputs=[Seg_Tracker, origin_frame, drawing_board, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side], outputs=[Seg_Tracker, input_first_frame, drawing_board])
        
        detect_button.click(fn=gd_detect, inputs=[Seg_Tracker, origin_frame, grounding_caption, box_threshold, text_threshold, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side], outputs=[Seg_Tracker, input_first_frame])
        
        track_for_video.click(fn=tracking_objects, inputs=[Seg_Tracker, input_video, input_img_seq, fps], outputs=[output_video, output_mask])
        
        output_res.select(fn=choose_obj_to_refine, inputs=[input_video, input_img_seq, Seg_Tracker, frame_num], outputs=[output_res, refine_idx])
        
        # Memory monitoring
        demo.load(fn=lambda: "Memory optimized version loaded", outputs=memory_status)
    
    return demo


def main():
    """Main function to launch the memory-optimized application"""
    demo = seg_track_app()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)


# Create demo object for import-based launching only when needed
demo = None

def get_demo():
    """Get or create the demo object"""
    global demo
    if demo is None:
        demo = seg_track_app()
    return demo

if __name__ == "__main__":
    main() 