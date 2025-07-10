from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "mamba2-130m")

# model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# 모델 불러오기 (Hugging Face Model Hub에서 바로 로드됨)
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba2-130m", trust_remote_code=True)

# 구조 출력
print(model)