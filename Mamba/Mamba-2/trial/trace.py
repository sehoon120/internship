from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("AntonV/mamba2-130m-hf")
input_ids = torch.tensor([[1, 42, 5123]])

def hook_fn(name):
    def hook(module, input, output):
        print(f"[{name}]")
        print(f"  input : {[x.shape for x in input]}")
        print(f"  output: {output.shape}")
    return hook

# hook 설치
for name, module in model.named_modules():
    if "mixer" in name or isinstance(module, torch.nn.Linear):
        module.register_forward_hook(hook_fn(name))

# 실행 시 shape 추적
with torch.no_grad():
    model(input_ids=input_ids)
