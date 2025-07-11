from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import os
from huggingface_hub import hf_hub_download
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "videomamba_tiny")

# 모델 체크포인트 다운로드
checkpoint_path = hf_hub_download(repo_id="OpenGVLab/VideoMamba", filename="videomamba_s16_in1k_res224.pth")

# 모델 config 다운로드
# config_path = hf_hub_download(repo_id="OpenGVLab/VideoMamba", filename="videomamba_s16_in1k_res224_config.yaml")

# config파일이 없어서 불가능