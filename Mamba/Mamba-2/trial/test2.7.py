import torch
import scipy.linalg
import os
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
model_path = os.path.join(base_dir, "mamba2-2.7b")

# === Step 1: Load Hugging Face Model Weights (FP32/FP16) ===
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("AntonV/mamba2-2.7b-hf")
model.save_pretrained(model_path + "/mamba2-2.7b-hf-raw", safe_serialization=False)
print('\n==================================================\n')
for name, param in model.named_parameters():
    print(name, param.shape)
print('\n==================================================\n')

# backbone.embeddings.weight torch.Size([50288, 2560])
# backbone.layers.63.norm.weight torch.Size([2560])
# backbone.layers.63.mixer.dt_bias torch.Size([80])
# backbone.layers.63.mixer.A_log torch.Size([80])
# backbone.layers.63.mixer.D torch.Size([80])
# backbone.layers.63.mixer.conv1d.weight torch.Size([5376, 1, 4])
# backbone.layers.63.mixer.conv1d.bias torch.Size([5376])
# backbone.layers.63.mixer.in_proj.weight torch.Size([10576, 2560])
# backbone.layers.63.mixer.norm.weight torch.Size([5120])
# backbone.layers.63.mixer.out_proj.weight torch.Size([2560, 5120])
# backbone.norm_f.weight torch.Size([2560])