import time
import torch
from transformers import AutoTokenizer
from mamba2 import Mamba2LMHeadModel
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, "../../..")
save_path = os.path.join(save_path, "mamba2-1.3b", "mamba2_1.3b_quantized.pth")
# save_path = os.path.join(save_path, "mamba2-2.7b/mamba2_2.7b_quantized.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES 적용 후 현재 장치:", torch.cuda.current_device())
print("사용 중인 장치 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))


def quantize_tensor_asym(tensor, num_bits=8):
    qmin = 0
    qmax = 2**num_bits - 1
    min_val = tensor.min()
    max_val = tensor.max()

    # scale and zero point
    scale = (max_val - min_val) / (qmax - qmin + 1e-8)
    zero_point = qmin - min_val / (scale + 1e-8)
    zero_point = torch.clamp(zero_point.round(), qmin, qmax)

    # quantize
    q_tensor = ((tensor / scale) + zero_point).round()
    q_tensor = torch.clamp(q_tensor, qmin, qmax)

    # dequantize
    dq_tensor = (q_tensor - zero_point) * scale
    return dq_tensor, scale, zero_point

def quantize_tensor_sym(tensor, num_bits=8):
    qmax = 2**(num_bits - 1) - 1
    scale = tensor.abs().max() / qmax
    q_tensor = (tensor / scale).round().clamp(-qmax, qmax)  # 실제 양자화값
    dq_tensor = q_tensor * scale
    return dq_tensor, scale

def quantize_tensor_per_channel_sym(tensor, num_bits=8, dim=0):
    qmax = 2 ** (num_bits - 1) - 1
    scales = tensor.abs().amax(dim=dim, keepdim=True) / qmax
    q_tensor = (tensor / scales).round().clamp(-qmax, qmax)
    dq_tensor = q_tensor * scales
    return dq_tensor, scales


model = Mamba2LMHeadModel.from_pretrained("state-spaces/mamba2-1.3b", device=device)
n_layers = len(model.backbone.layers)

for i in range(n_layers):
    layer = model.backbone.layers[i]
    mixer = layer['mixer']
    norm = layer['norm']

    with torch.no_grad():
        # 전체 weight 양자화 (row가 아닌 전체)
        q, _, zp = quantize_tensor_asym(mixer.in_proj.weight, 8)
        mixer.in_proj.weight.copy_(q)

        q, _ = quantize_tensor_per_channel_sym(mixer.conv1d.weight, 8)
        mixer.conv1d.weight.copy_(q)

        q, _ = quantize_tensor_sym(mixer.conv1d.bias, 8)
        mixer.conv1d.bias.copy_(q)

        q, _ = quantize_tensor_sym(mixer.A_log, 16)
        mixer.A_log.copy_(q)

        q, _ = quantize_tensor_sym(mixer.D, 8)
        mixer.D.copy_(q)

        q, _ = quantize_tensor_sym(mixer.dt_bias, 16)
        mixer.dt_bias.copy_(q)

        q, _ = quantize_tensor_sym(mixer.norm.weight, 16)
        mixer.norm.weight.copy_(q)

        q, _, zp = quantize_tensor_asym(mixer.out_proj.weight, 8)
        mixer.out_proj.weight.copy_(q)

        q, _ = quantize_tensor_sym(norm.weight, 16)
        norm.weight.copy_(q)

# ▷ 레이어 외부 RMSNorm
# with torch.no_grad():
    # q_param, scale = quantize_tensor_sym(norm.weight, num_bits=8)
    # norm.weight.copy_(q_param)

# 최종 저장
torch.save(model.state_dict(), save_path)


'''
# 저장한 가중치 다시 부르기
# config는 원래대로 불러와야 함
config = Mamba2Config(d_model=768, n_layer=24, vocab_size=50277)
model = Mamba2LMHeadModel(config)

# 수정된 가중치 로드
state_dict = torch.load("mamba2_130m_modified.pth")
model.load_state_dict(state_dict)
model.eval()
'''
