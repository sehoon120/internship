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
save_path = os.path.join(model_dir, "mamba2_2.7b_quantized.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("CUDA_VISIBLE_DEVICES 적용 후 현재 장치:", torch.cuda.current_device())
# print("사용 중인 장치 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))


def quantize_tensor_asym(tensor, num_bits=8, frac = 6):
    qmin = 0
    qmax = 2**num_bits - 1
    min_val = tensor.min()

    # scale and zero point
    scale = 1 / (2**frac)  # 0.015625
    zero_point = qmin - min_val / (scale)
    zero_point = torch.clamp(zero_point.round(), qmin, qmax).int()

    # quantize
    q_tensor = torch.round(tensor / scale + zero_point).clamp(qmin, qmax).to(torch.uint8)

    # dequantize
    dq_tensor = (q_tensor.to(torch.float32) - zero_point) * scale
    return dq_tensor, scale, zero_point
    # return q_tensor, scale, zero_point

def quantize_tensor_sym(tensor, num_bits=8):
    qmax = 2**(num_bits - 1) - 1
    scale = tensor.abs().max() / qmax
    q_tensor = (tensor / scale).round().clamp(-qmax, qmax)  # 실제 양자화값
    dq_tensor = q_tensor * scale
    # return dq_tensor, scale
    return q_tensor, scale

def quantize_tensor_per_channel_sym(tensor, num_bits=8, dim=0):
    qmax = 2 ** (num_bits - 1) - 1
    scales = tensor.abs().amax(dim=dim, keepdim=True) / qmax
    q_tensor = (tensor / scales).round().clamp(-qmax, qmax)
    dq_tensor = q_tensor * scales
    # return dq_tensor, scales
    return q_tensor, scales


# model = Mamba2LMHeadModel.from_pretrained("state-spaces/mamba2-1.3b", device=device)
model = Mamba2LMHeadModel.from_pretrained("state-spaces/mamba2-2.7b", device=device)
n_layers = len(model.backbone.layers)
quant_metadata = {}

for i in range(n_layers):
    layer = model.backbone.layers[i]
    mixer = layer['mixer']
    norm = layer['norm']

    with torch.no_grad():
        q, sc, zp = quantize_tensor_asym(mixer.in_proj.weight, 8)  # 8.6
        mixer.in_proj.weight.copy_(q)
        quant_metadata[f"layers.{i}.mixer.in_proj.weight"] = {"scale": sc, "zero_point": zp}
        # print(f"{i} | mixer.in_proj.weight | sc: {sc}, zp: {zp}")
        q, sc = fxp8_6.dequantize(fxp8_6.quantize(mixer.conv1d.weight)), fxp8_6.scale
        mixer.conv1d.weight.copy_(q)
        # frac_bits = int(-torch.log2(sc.max()).item())
        # fxp_format = f"FXP8.{frac_bits}"
        # fabric_bits = 8 - frac_bits
        quant_metadata[f"layers.{i}.mixer.conv1d.weight"] = {"scale": sc}
        # print(f"{i} | mixer.conv1d.weight | {fxp_format}")
        q, sc = fxp8_4.dequantize(fxp8_4.quantize(mixer.conv1d.bias)), fxp8_4.scale
        mixer.conv1d.bias.copy_(q)
        quant_metadata[f"layers.{i}.mixer.conv1d.bias"] = {"scale": sc}
        # print(f"{i} | mixer.conv1d.bias | sc: {sc}")
        q, sc = fxp16_12.dequantize(fxp16_12.quantize(mixer.A_log)), fxp16_12.scale  # 13 -> 12
        mixer.A_log.copy_(q)
        quant_metadata[f"layers.{i}.mixer.A_log"] = {"scale": sc}
        # print(f"{i} | mixer.A_log | sc: {sc}")
        q, sc = fxp8_4.dequantize(fxp8_4.quantize(mixer.D)), fxp8_4.scale  # 5 -> 4
        mixer.D.copy_(q)
        quant_metadata[f"layers.{i}.mixer.D"] = {"scale": sc}
        # print(f"{i} | mixer.D | sc: {sc}")
        q, sc = fxp16_11.dequantize(fxp16_11.quantize(mixer.dt_bias)), fxp16_11.scale  # 12 -> 11
        mixer.dt_bias.copy_(q)
        quant_metadata[f"layers.{i}.mixer.dt_bias"] = {"scale": sc}
        # print(f"{i} | mixer.dt_bias | sc: {sc}")
        q, sc = fxp16_11.dequantize(fxp16_11.quantize(mixer.norm.weight)), fxp16_11.scale  # 13 -> 11
        mixer.norm.weight.copy_(q)
        quant_metadata[f"layers.{i}.mixer.norm.weight"] = {"scale": sc}
        # print(f"{i} | mixer.norm.weight | sc: {sc}")
        q, sc, zp = quantize_tensor_asym(mixer.out_proj.weight, frac=5)  # 8.6 -> 8.5
        mixer.out_proj.weight.copy_(q)
        quant_metadata[f"layers.{i}.mixer.out_proj.weight"] = {"scale": sc, "zero_point": zp}
        # print(f"{i} | mixer.out_proj.weight | sc: {sc}, zp: {zp}")
        q, sc = fxp16_13.dequantize(fxp16_13.quantize(norm.weight)), fxp16_13.scale  # 14 -> 13
        norm.weight.copy_(q)
        quant_metadata[f"layers.{i}.norm.weight"] = {"scale": sc}
        # print(f"{i} | norm.weight | sc: {sc}")

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
