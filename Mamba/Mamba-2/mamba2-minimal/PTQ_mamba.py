'''
    mamba2-130m quantization FXP8/16 standard

    mamba block 정의
'''

# 필요한 라이브러리 import
import time
import logging
import torch
from torch import nn
from transformers import AutoTokenizer
from mamba2 import Mamba2LMHeadModel, Mamba2Config, ssd, InferenceCache
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# print("Current device:", torch.cuda.current_device())
# print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# print("Allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
# print("Reserved :", torch.cuda.memory_reserved() / 1024**3, "GB")

current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, "log")
model_path = os.path.join(current_dir, '../../..')
model_path = os.path.join(model_path, 'mamba2-2.7b/mamba2_2.7b_quantized.pth')

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

def q_dq(x, a, b):
    if a == 8:
        if b == 2:
            return fxp8_2.dequantize(fxp8_2.quantize(x))
        elif b == 3:
            return fxp8_3.dequantize(fxp8_3.quantize(x))
        elif b == 4:
            return fxp8_4.dequantize(fxp8_4.quantize(x))
        elif b == 5:
            return fxp8_5.dequantize(fxp8_5.quantize(x))
        elif b == 6:
            return fxp8_6.dequantize(fxp8_6.quantize(x))
        elif b == 7:
            return fxp8_7.dequantize(fxp8_7.quantize(x))

    elif a == 16:
        if b == 4:
            return fxp16_4.dequantize(fxp16_4.quantize(x))
        elif b == 5:
            return fxp16_5.dequantize(fxp16_5.quantize(x))
        elif b == 6:
            return fxp16_6.dequantize(fxp16_6.quantize(x))
        elif b == 7:
            return fxp16_7.dequantize(fxp16_7.quantize(x))
        elif b == 8:
            return fxp16_8.dequantize(fxp16_8.quantize(x))
        elif b == 9:
            return fxp16_9.dequantize(fxp16_9.quantize(x))
        elif b == 10:
            return fxp16_10.dequantize(fxp16_10.quantize(x))
        elif b == 11:
            return fxp16_11.dequantize(fxp16_11.quantize(x))
        elif b == 12:
            return fxp16_12.dequantize(fxp16_12.quantize(x))
        elif b == 13:
            return fxp16_13.dequantize(fxp16_13.quantize(x))
        elif b == 14:
            return fxp16_14.dequantize(fxp16_14.quantize(x))
        elif b == 15:
            return fxp16_15.dequantize(fxp16_15.quantize(x))

    elif a == 32:
        if b == 16:
            return fxp32_16.dequantize(fxp32_16.quantize(x))
        elif b == 18:
            return fxp32_18.dequantize(fxp32_18.quantize(x))
        elif b == 20:
            return fxp32_20.dequantize(fxp32_20.quantize(x))
        elif b == 21:
            return fxp32_21.dequantize(fxp32_21.quantize(x))
        elif b == 24:
            return fxp32_24.dequantize(fxp32_24.quantize(x))

    raise ValueError(f"Unsupported FXP format: {a} total bits with {b} fractional bits")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config = Mamba2Config(d_model=768, n_layer=24, vocab_size=50277)
config = Mamba2Config(d_model=2560, n_layer=64, vocab_size=50288)

model = Mamba2LMHeadModel(config)
# model.load_state_dict(torch.load(r"C:\Internship\mamba2-130m\mamba2_130m_quantized.pth"))
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# model_name = 'state-spaces/mamba2-2.7b'  # "AntonV/mamba2-130m-hf"
# model = Mamba2LMHeadModel.from_pretrained(model_name, device=device)


def Mamba_Block(model, config, residual, h, i):
    # Norm & Quant
    x = model.backbone['layers'][i].norm(residual)  # RMSNorm
    x = q_dq(x, 16, 11)  # -3.2 ~ 4.6

    # Linear Projection
    zxbcdt = model.backbone['layers'][i]['mixer'].in_proj(x.squeeze(1))
    z, xBC, dt = torch.split(
        zxbcdt,
        [config.d_inner, config.d_inner + 2 * config.d_state, config.nheads],
        dim=-1,
    )
    xBC = q_dq(xBC, 16, 11)
    z = q_dq(z, 16, 11)
    dt = q_dq(dt, 16, 11)

    # Convolution
    h[i].conv_state.copy_(torch.roll(h[i].conv_state, shifts=-1, dims=-1))
    h[i].conv_state[:, :, -1] = xBC
    xBC = torch.sum(
        h[i].conv_state * rearrange(model.backbone['layers'][i]['mixer'].conv1d.weight, "d 1 w -> d w"),
        dim=-1
    )
    xBC += model.backbone['layers'][i]['mixer'].conv1d.bias
    xBC = q_dq(xBC, 16, 11)
    xBC = F.silu(xBC)
    xBC = q_dq(xBC, 16, 10)

    # Split x, B, C
    x, B, C = torch.split(xBC, [config.d_inner, config.d_state, config.d_state], dim=-1)
    x = q_dq(x, 16, 10)
    B = q_dq(B, 16, 10) if i == 19 else q_dq(B, 16, 12)
    C = q_dq(C, 16, 11)

    # A, dt, dA
    A = -torch.exp(model.backbone['layers'][i]['mixer'].A_log)
    A = q_dq(A, 32, 16)
    dt = F.softplus(dt + model.backbone['layers'][i]['mixer'].dt_bias)
    dt = q_dq(dt, 16, 11)
    dA = dt * A
    dA = q_dq(dA, 32, 16)
    dA = torch.exp(dA)
    dA = q_dq(dA, 16, 14)

    # State Update
    x = rearrange(x, "b (h p) -> b h p", p=config.headdim)
    dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
    dBx = q_dq(dBx, 16, 11)
    dAh = h[i].ssm_state * rearrange(dA, "b h -> b h 1 1")
    dAh = q_dq(dAh, 16, 10)
    dAhdBx = dAh + dBx
    dAhdBx = q_dq(dAhdBx, 16, 10)
    h[i].ssm_state.copy_(dAhdBx)

    # Output computation
    y = torch.einsum("bhpn, bn -> bhp", h[i].ssm_state, C)
    y = q_dq(y, 16, 7)
    y = y + rearrange(model.backbone['layers'][i]['mixer'].D, "h -> h 1") * x
    y = q_dq(y, 16, 7)

    # Apply SiLU(z)
    z = F.silu(z)
    z = q_dq(z, 16, 11)
    y = y * z
    y = q_dq(y, 32, 16)

    # Final norm + linear proj
    y = rearrange(y, "b h p -> b (h p)")
    y = model.backbone['layers'][i]['mixer'].norm(y)
    y = q_dq(y, 32, 16)
    y = model.backbone['layers'][i]['mixer'].out_proj(y)
    y = q_dq(y, 32, 16)

    # Residual update
    residual = residual + y.unsqueeze(1)
    return residual, h
