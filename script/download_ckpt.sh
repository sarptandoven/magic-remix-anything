#!/bin/bash

# Create necessary directories
mkdir -p ./ckpt
mkdir -p ./ast_master/pretrained_models

# download aot-ckpt 
gdown '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ' --output ./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth

# download sam-ckpt
curl -L -o ./ckpt/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# download grounding-dino ckpt
curl -L -o ./ckpt/groundingdino_swint_ogc.pth https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth

# download mit-ast-finetuned ckpt
curl -L -o ./ast_master/pretrained_models/audio_mdl.pth https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1