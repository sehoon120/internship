import time
import torch
from transformers import AutoTokenizer
from mamba2 import Mamba2LMHeadModel
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from FXP_simulator import FXP16Simulator, FXP32Simulator, FXP8Simulator
# 8-bit FXP 시뮬레이터
fxp8_2 = FXP8Simulator(frac_bits=2)
fxp8_3 = FXP8Simulator(frac_bits=3)
fxp8_4 = FXP8Simulator(frac_bits=4)
fxp8_5 = FXP8Simulator(frac_bits=5)
fxp8_6 = FXP8Simulator(frac_bits=6)
fxp8_7 = FXP8Simulator(frac_bits=7)
# 16-bit FXP 시뮬레이터
fxp16_4 = FXP16Simulator(frac_bits=4)
fxp16_5 = FXP16Simulator(frac_bits=5)
fxp16_6 = FXP16Simulator(frac_bits=6)
fxp16_7 = FXP16Simulator(frac_bits=7)
fxp16_8 = FXP16Simulator(frac_bits=8)
fxp16_9 = FXP16Simulator(frac_bits=9)
fxp16_10 = FXP16Simulator(frac_bits=10)
fxp16_11 = FXP16Simulator(frac_bits=11)
fxp16_12 = FXP16Simulator(frac_bits=12)
fxp16_13 = FXP16Simulator(frac_bits=13)
fxp16_14 = FXP16Simulator(frac_bits=14)
fxp16_15 = FXP16Simulator(frac_bits=15)
# 32-bit FXP 시뮬레이터
fxp32_16 = FXP32Simulator(frac_bits=16)
fxp32_18 = FXP32Simulator(frac_bits=18)
fxp32_20 = FXP32Simulator(frac_bits=20)
fxp32_21 = FXP32Simulator(frac_bits=21)
fxp32_24 = FXP32Simulator(frac_bits=24)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "../../../..", "mamba2-2.7b")
os.makedirs(model_dir, exist_ok=True)  # 디렉토리 없으면 생성
save_path = os.path.join(model_dir, "mamba2_2.7b_quantized_FP.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("CUDA_VISIBLE_DEVICES 적용 후 현재 장치:", torch.cuda.current_device())
# print("사용 중인 장치 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))


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


model = Mamba2LMHeadModel.from_pretrained("state-spaces/mamba2-2.7b", device=device)
n_layers = len(model.backbone.layers)

for i in range(n_layers):
    layer = model.backbone.layers[i]
    mixer = layer['mixer']
    norm = layer['norm']

    with torch.no_grad():
        # 전체 weight 양자화 (row가 아닌 전체)
        q, _, zp = quantize_tensor_asym(mixer.in_proj.weight, 8)
        mixer.in_proj.weight.copy_(q)

        # q, _ = quantize_tensor_sym(mixer.conv1d.weight, 8)
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

# int8/int16 그대로 저장
# torch.save({
#     "model_int8": {
#         k: v.to(torch.int8) if 'weight' in k or 'bias' in k else v
#         for k, v in model.state_dict().items()
#     },
#     "quant_metadata": quant_metadata
# }, save_path)
