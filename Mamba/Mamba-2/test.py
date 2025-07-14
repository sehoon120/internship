import torch
import scipy.linalg
import os
from transformers import AutoTokenizer, AutoConfig, register_auto_class
from mamba_ssm.models.mamba2 import MambaLMHeadModel

# ✅ HuggingFace에 커스텀 모델 클래스 등록
register_auto_class(MambaLMHeadModel)

# 🔁 로딩
tokenizer = AutoTokenizer.from_pretrained("AntonV/mamba2-130m-hf")
config = AutoConfig.from_pretrained("AntonV/mamba2-130m-hf", trust_remote_code=True)
model = MambaLMHeadModel.from_pretrained("AntonV/mamba2-130m-hf", config=config, trust_remote_code=True)


current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "mamba2-130m")

# === Step 1: Load Hugging Face Model Weights (FP32/FP16) ===
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("AntonV/mamba2-130m-hf")
model.save_pretrained(model_path + "/mamba2-130m-hf-raw", safe_serialization=False)



weight = model.lm_head.weight.data.clone()