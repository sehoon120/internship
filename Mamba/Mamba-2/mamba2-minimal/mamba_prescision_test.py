'''
    mamba2-130m quantization FXP8/16 standard

    Fully FXP operation으로 변경한 모델
'''

# 필요한 라이브러리 import
import time
import logging
import torch
from torch import nn
from transformers import AutoTokenizer
from mamba2 import Mamba2LMHeadModel, Mamba2Config, ssd, InferenceCache
from SiLU_graph import ApproxSiLU16_FXP
from exp_graph import ApproxExp16_FXP, ApproxExp_FXP32in16out14
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
model_dir = os.path.join(current_dir, "../../../..", "mamba2-2.7b")
os.makedirs(model_dir, exist_ok=True)  # 디렉토리 없으면 생성
model_path = os.path.join(model_dir, "mamba2_2.7b_quantized_FXP.pth")

log_dir = os.path.join(current_dir, "log")

os.makedirs(log_dir, exist_ok=True)
# 로거 생성
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 파일 핸들러
file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'), mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
# 콘솔 핸들러
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
# 핸들러 등록
logger.addHandler(file_handler)
# logger.addHandler(console_handler)

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

def fxp_multiply_and_cast_to_fxp16_11(x_fxp32_16, w_fxp16_13, out_bit = 16, out_frac = 11):
    # 1. 정수 곱셈 (int64로 overflow 방지)
    product = x_fxp32_16.to(torch.int64) * w_fxp16_13.to(torch.int64)  # FXP64.29

    # 2. shift right to get FXP16.11
    result = (product >> (29 - out_frac)).clamp(-2**(out_bit-1), 2**(out_bit-1) - 1)  # FXP16.11 범위
    return result.to(torch.int16)

def fxp_linear_in_proj(x_fxp16_11, W_qint8, scale, zp, in_frac=11, w_frac=6, out_frac=11):
    # x: (B, D_in), W: (D_out, D_in)
    # B, D_in = x_fxp16_11.shape
    # D_out = W_qint8.shape[0]
    
    # 1. Expand dimensions for broadcasting
    x_int = (x_fxp16_11 * (1 << in_frac)).round().to(torch.int32).cpu()  # B x D_in
    W_int = (W_qint8.to(torch.int32) - zp).cpu()  # D_out x D_in

    # 2. Integer matrix multiplication
    # shape: (B, D_out)
    acc = torch.matmul(x_int, W_int.T)  # int32 결과

    # 3. Scale compensation
    # total fixed point product: in_frac + w_frac bits → need to match out_frac
    total_frac = in_frac + w_frac  # = 17
    scale_factor = scale * (1 << w_frac)  # float scale * 2^w_frac

    if out_frac >= total_frac:
        scale_int = int(round(scale_factor * (1 << (out_frac - total_frac))))
    else:
        scale_int = int(round(scale_factor / (1 << (total_frac - out_frac))))
    # scale_int = int(round(scale_factor * (1 << (out_frac - total_frac))))  # rescale to FXP out_frac

    # 4. Final fixed-point scaling
    out = (acc * scale_int + (1 << (out_frac - 1))) >> out_frac  # rounding
    out = torch.clamp(out, -2**15, 2**15 - 1).to(torch.int16)  # clamp to int16

    # 5. Convert back to FXP16.11 float form
    return (out.to(torch.float32) / (1 << out_frac)).to(x_fxp16_11.device)

def fxp_conv1d_step(conv_state_fxp611, weight_qint8_fx8_6, bias_fx8_4, in_frac=11, w_frac=6, bias_frac=4, out_frac=11):
    """
    conv_state_fxp611: (D, W)   -- int16 or float FXP6.11
    weight_qint8_fx8_6: (D, W)  -- int8 or int32, FXP8.6
    bias_fx8_4: (D,)            -- float or int, FXP8.4

    Returns:
        xBC: (D,) -- float32, FXP16.11 simulated
    """

    # Step 1: quantize input to int
    x_int = (conv_state_fxp611 * (1 << in_frac)).round().to(torch.int32)  # (D, W)
    w_int = (weight_qint8_fx8_6 * (1 << w_frac)).round().to(torch.int32)  # (D, W)

    # Step 2: pointwise multiply and sum over width dim
    acc = torch.sum(x_int * w_int, dim=-1)  # (D,) int32

    # Step 3: apply bias
    bias_int = (bias_fx8_4 * (1 << bias_frac)).round().to(torch.int32)  # FXP8.4 → int
    acc += bias_int << (in_frac + w_frac - bias_frac)  # align to total_frac

    # Step 4: rescale to FXP16.11 (out_frac)
    total_frac = in_frac + w_frac  # 17
    if out_frac <= total_frac:
        x_int = (acc + (1 << (total_frac - out_frac - 1))) >> (total_frac - out_frac)
    else:
        x_int = acc << (out_frac - total_frac)

    # Step 5: clamp to int16 and return float
    x_int = torch.clamp(x_int, -2**15, 2**15 - 1).to(torch.int16)
    return x_int.to(torch.float32) / (1 << out_frac)

class ApproxLog1p_FXP(nn.Module):
    def __init__(self, in_frac=16, out_frac=11):
        super().__init__()
        self.in_frac = in_frac
        self.out_frac = out_frac

        # 입력 범위: y ∈ [0, ~60]
        self.x_pts_fp = torch.linspace(0, 64.0, steps=17)  # 16 segments
        self.log1p_vals_fp = torch.log1p(self.x_pts_fp)

        self.x_pts = (self.x_pts_fp * (1 << in_frac)).round().to(torch.int64)
        self.y_vals = (self.log1p_vals_fp * (1 << out_frac)).round().to(torch.int64)

    def forward(self, y):  # y is FXP.int float
        y_int = (y * (1 << self.in_frac)).round().to(torch.int64)
        out_int = torch.empty_like(y_int)

        min_x, max_x = self.x_pts[0].item(), self.x_pts[-1].item()
        mask_low = y_int <= min_x
        mask_high = y_int >= max_x
        y_clamped = torch.clamp(y_int, min_x, max_x)

        out_int[mask_low] = self.y_vals[0]
        out_int[mask_high] = self.y_vals[-1]

        idx = torch.bucketize(y_clamped, self.x_pts.to(y.device)) - 1
        idx = torch.clamp(idx, 0, len(self.x_pts) - 2)

        x0 = self.x_pts.to(y.device)[idx]
        x1 = self.x_pts.to(y.device)[idx + 1]
        y0 = self.y_vals.to(y.device)[idx]
        y1 = self.y_vals.to(y.device)[idx + 1]

        dx = y_clamped - x0
        dx_total = x1 - x0
        dx_total = torch.clamp(dx_total, min=1)

        t_fx = ((dx << self.out_frac) + (dx_total // 2)) // dx_total
        dy = y1 - y0
        interp = y0 + ((t_fx * dy + (1 << (self.out_frac - 1))) >> self.out_frac)

        out_int[~(mask_low | mask_high)] = interp[~(mask_low | mask_high)]
        return out_int.to(torch.float32) / (1 << self.out_frac)

class ApproxSoftplus_ExpLog1p(nn.Module):
    def __init__(self, in_frac=12, mid_frac=16, out_frac=11):
        super().__init__()
        self.in_frac = in_frac
        self.exp = ApproxExp16_FXP(in_frac=in_frac, out_frac=mid_frac)
        self.log1p = ApproxLog1p_FXP(in_frac=mid_frac, out_frac=out_frac)

    def forward(self, x):  # x: FXP.float
        exp_x = self.exp(x)       # float (FXP32.16)
        log1p_x = self.log1p(exp_x)
        return log1p_x  # float (FXP16.11)

def fxp_mul_dt_A(dt_fxp11, A_fxp13, dt_frac=11, A_frac=13, out_frac=16):
    dt_int = (dt_fxp11 * (1 << dt_frac)).round().to(torch.int64)  # 64bit for safety
    A_int = (A_fxp13 * (1 << A_frac)).round().to(torch.int64)

    acc = dt_int * A_int  # FXP(dt+frac + A_frac) = FXP24

    shift = dt_frac + A_frac - out_frac  # 11 + 13 - 16 = 8
    dA_int = (acc + (1 << (shift - 1))) >> shift  # rounding shift

    return dA_int.to(torch.float32) / (1 << out_frac)  # back to FXP32.16 float

def fxp_mul3_dtBx(dt, B, x, dt_frac=11, B_frac=10, x_frac=10, out_frac=11):
    """
    dt: (B, H)      FXP16.11
    B:  (B, N)      FXP16.10
    x:  (B, H, P)   FXP16.10
    Return:
        dBx: (B, H, P, N)  FXP16.11
    """
    B_, H, P = x.shape
    _, N = B.shape

    # Step 1: FXP quantize to int
    dt_int = (dt * (1 << dt_frac)).round().to(torch.int64).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
    B_int = (B * (1 << B_frac)).round().to(torch.int64).unsqueeze(1).unsqueeze(2)      # (B, 1, 1, N)
    x_int = (x * (1 << x_frac)).round().to(torch.int64).unsqueeze(-1)                  # (B, H, P, 1)

    acc = dt_int * B_int * x_int  # (B, H, P, N), FXP(31)

    shift = dt_frac + B_frac + x_frac - out_frac  # 31 - 11 = 20
    acc_shifted = (acc + (1 << (shift - 1))) >> shift  # rounding shift

    # Clamp to int16 and dequantize
    acc_clamped = torch.clamp(acc_shifted, -2**15, 2**15 - 1).to(torch.int16)
    return acc_clamped.to(torch.float32) / (1 << out_frac)

def fxp_mul_dAh(h_ssm_state, dA, h_frac=10, dA_frac=14, out_frac=10):
    """
    h_ssm_state: (B, H, N, D)   FXP16.10
    dA         : (B, H)         FXP16.14
    Return:
        dAh: (B, H, N, D)       FXP16.10
    """
    h_int = (h_ssm_state * (1 << h_frac)).round().to(torch.int64)  # (B, H, N, D)
    dA_int = (dA * (1 << dA_frac)).round().to(torch.int64).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

    acc = h_int * dA_int  # FXP24

    shift = h_frac + dA_frac - out_frac  # 24
    acc_shifted = (acc + (1 << (shift - 1))) >> shift  # rounding

    return torch.clamp(acc_shifted, -2**15, 2**15 - 1).to(torch.int16).to(torch.float32) / (1 << out_frac)

def fxp_add_dAh_dBx(dAh, dBx, dAh_frac=10, dBx_frac=11, out_frac=10):
    """
    dAh: FXP16.10 (float)
    dBx: FXP16.11 (float)
    Return: FXP16.10 (float)
    """
    dAh_int = (dAh * (1 << dAh_frac)).round().to(torch.int32)
    dBx_int = (dBx * (1 << dBx_frac)).round().to(torch.int32)

    shift = dBx_frac - out_frac  # = 1
    dBx_aligned = dBx_int >> shift  # FXP16.11 → FXP16.10

    dSum_int = dAh_int + dBx_aligned
    dSum_clamped = torch.clamp(dSum_int, -2**15, 2**15 - 1).to(torch.int16)

    return dSum_clamped.to(torch.float32) / (1 << out_frac)

def fxp_dot_ssm_C(ssm_state, C, ssm_frac=10, C_frac=10, out_frac=7):
    """
    ssm_state: (B, H, P, N)   FXP16.10
    C        : (B, N)         FXP16.10
    Return:
        y: (B, H, P)          FXP16.7
    """
    B, H, P, N = ssm_state.shape
    ssm_int = (ssm_state * (1 << ssm_frac)).round().to(torch.int64)  # (B, H, P, N)
    C_int = (C * (1 << C_frac)).round().to(torch.int64)              # (B, N)

    # reshape for broadcasting
    C_int = C_int.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N)

    acc = (ssm_int * C_int).sum(dim=-1)  # sum over N → (B, H, P), FXP(ssm+C = 20)

    shift = ssm_frac + C_frac - out_frac  # = 20 - 7 = 13
    acc_shifted = (acc + (1 << (shift - 1))) >> shift  # rounding shift

    y_int = torch.clamp(acc_shifted, -2**15, 2**15 - 1).to(torch.int16)
    return y_int.to(torch.float32) / (1 << out_frac)

def fxp_add_Dx_to_y(y, x, D, x_frac=10, D_frac=5, y_frac=7):
    """
    y: (B, H, P)     FXP16.7
    x: (B, H, P)     FXP16.10
    D: (H,)          FXP8.5
    Return:          FXP16.7
    """
    y_int = (y * (1 << y_frac)).round().to(torch.int32)
    x_int = (x * (1 << x_frac)).round().to(torch.int32)
    D_int = (D * (1 << D_frac)).round().to(torch.int32).unsqueeze(0).unsqueeze(-1)  # (1, H, 1)

    Dx_int = D_int * x_int  # FXP15
    shift = D_frac + x_frac - y_frac  # 15 - 7 = 8
    Dx_aligned = (Dx_int + (1 << (shift - 1))) >> shift  # rounding shift

    y_sum_int = y_int + Dx_aligned
    y_sum_clamped = torch.clamp(y_sum_int, -2**15, 2**15 - 1).to(torch.int16)

    return y_sum_clamped.to(torch.float32) / (1 << y_frac)

def fxp_mul_y_z_to_32_16(y, z, y_frac=7, z_frac=11, out_frac=16):
    """
    y: (B, H, P)     FXP16.7
    z: (B, H, P)     FXP16.11
    return:          FXP32.16
    """
    y_int = (y * (1 << y_frac)).round().to(torch.int64)
    z_int = (z * (1 << z_frac)).round().to(torch.int64)

    acc = y_int * z_int  # FXP32.18
    shift = y_frac + z_frac - out_frac  # 18 - 16 = 2
    acc_shifted = (acc + (1 << (shift - 1))) >> shift  # rounding

    return acc_shifted.to(torch.float32) / (1 << out_frac)  # float32지만 의미는 FXP32.16

def fxp_linear_out_proj_32in_32out(x_fxp32_16, W_qint8, scale, zp, in_frac=16, w_frac=6, out_frac=16):
    """
    x_fxp32_16: (B, D_in) float32, 의미상 FXP32.16
    W_qint8:   (D_out, D_in) torch.int8, zero-centered quant
    scale:     float
    zp:        int
    return:    float32 (FXP32.16 의미)
    """
    B, D_in = x_fxp32_16.shape
    D_out = W_qint8.shape[0]

    # 1. 정수 변환
    x_int = (x_fxp32_16 * (1 << in_frac)).round().to(torch.int64).cpu()  # FXP32.16 → int
    W_int = (W_qint8.to(torch.int64) - zp).cpu()  # int8 → zero-centered

    # 2. 행렬 곱 (B, D_out)
    acc = torch.matmul(x_int, W_int.T)  # 결과: FXP(in + w) = FXP22

    # 3. scale factor 적용
    total_frac = in_frac + w_frac  # 16 + 6 = 22
    scale_factor = scale * (1 << w_frac)

    # 정수 scaling 계수로 변환
    if out_frac >= total_frac:
        scale_int = int(round(scale_factor * (1 << (out_frac - total_frac))))
    else:
        scale_int = int(round(scale_factor / (1 << (total_frac - out_frac))))

    # 4. 최종 스케일 적용 및 정렬
    acc_scaled = (acc * scale_int + (1 << (out_frac - 1))) >> out_frac  # rounding

    # 5. clamp + float 변환
    acc_clamped = torch.clamp(acc_scaled, -2**31, 2**31 - 1).to(torch.int32)
    return (acc_clamped.to(torch.float32) / (1 << out_frac)).to(x_fxp32_16.device)  # FXP32.16 의미


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config = Mamba2Config(d_model=768, n_layer=24, vocab_size=50277)
config = Mamba2Config(d_model=2560, n_layer=64, vocab_size=50288)

model = Mamba2LMHeadModel(config)
ckpt = torch.load(model_path, map_location=device)
# model.load_state_dict(torch.load(r"C:\Internship\mamba2-130m\mamba2_130m_quantized.pth"))
model.load_state_dict(ckpt["model_int8"])
model = model.to(device)
quant_info = ckpt["quant_metadata"]

model_name = 'state-spaces/mamba2-2.7b'  # "AntonV/mamba2-130m-hf"
model_FP = Mamba2LMHeadModel.from_pretrained(model_name, device=device)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id
h = [InferenceCache.alloc(
    batch_size=1,
    args=config,
    device=device
) for _ in range(config.n_layer)]

# prompt = """
# Mamba is a new sequence model that can replace transformers in some cases. 
# It uses state space models instead of attention. Its advantage is that it is faster and more memory-efficient.
# Write a clear summary of how Mamba differs from Transformers.
# """
# prompt = """
# John has 3 apples. He gives 1 to Mary and buys 4 more. How many apples does he have now?
# """
prompt = """
Continue the story: "The robot slowly opened the door, not knowing what it would find on the other side..."
"""
# prompt = """
# Write a Python function that returns the nth Fibonacci number using recursion.
# """
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)  # shape: (1, L)

prefix, f_token = input_ids[:, :-1], input_ids[:, -1:]
chunk_size = model_FP.args.chunk_size
n_chunked = (prefix.shape[1] // chunk_size) * chunk_size
if n_chunked > 0:
    _, h = model_FP(prefix[:, :n_chunked], None)
else:
    h = [InferenceCache.alloc(1, model_FP.args, device=device) for _ in range(model_FP.args.n_layer)]
for i in range(n_chunked, prefix.shape[1]):
    _, h = model_FP(prefix[:, i:i+1], h)

generated = [t.item() for t in input_ids[0]]

entrophy_list = []
loss_list = []
eps = 1e-5
silu_fxp_11to10 = ApproxSiLU16_FXP(in_frac=11, out_frac=10)
silu_fxp_11to11 = ApproxSiLU16_FXP(in_frac=11, out_frac=11)
approx_exp = ApproxExp16_FXP(in_frac=13, out_frac=16)
approx_softplus = ApproxSoftplus_ExpLog1p(in_frac=12, mid_frac=16, out_frac=11)
approx_exp_32 = ApproxExp_FXP32in16out14(in_frac=16, out_frac=14)            


'''
# x1_list = [[] for _ in range(config.n_layer)] 
# xBC1_list = [[] for _ in range(config.n_layer)] 
# z1_list = [[] for _ in range(config.n_layer)] 
# dt1_list = [[] for _ in range(config.n_layer)] 
# xBC2_list = [[] for _ in range(config.n_layer)] 
# xBC3_list = [[] for _ in range(config.n_layer)] 
# x2_list = [[] for _ in range(config.n_layer)] 
# B_list = [[] for _ in range(config.n_layer)] 
# C_list = [[] for _ in range(config.n_layer)] 
# A_list = [[] for _ in range(config.n_layer)] 
# dt2_list = [[] for _ in range(config.n_layer)] 
# dA1_list = [[] for _ in range(config.n_layer)] 
# dA2_list = [[] for _ in range(config.n_layer)] 
# dBx_list = [[] for _ in range(config.n_layer)] 
# dAh_list = [[] for _ in range(config.n_layer)] 
# dAhdBx_list = [[] for _ in range(config.n_layer)] 
# y1_list = [[] for _ in range(config.n_layer)] 
# y2_list = [[] for _ in range(config.n_layer)] 
# z2_list = [[] for _ in range(config.n_layer)] 
# y3_list = [[] for _ in range(config.n_layer)] 
# y4_list = [[] for _ in range(config.n_layer)] 
# y5_list = [[] for _ in range(config.n_layer)] 
# residual_list = [[] for _ in range(config.n_layer)] 
'''

with torch.no_grad():
    for t in range(10):  # 100
        seqlen = f_token.shape[1]
        input_tensor = f_token.to(device)
        u = model.backbone['embedding'](input_tensor)
        residual = u  
        residual = fxp32_16.quantize(residual)
        # residual = q_dq(residual, 32, 16)  # -2.593 ~ 3.061
        # 이거 하니까 엔트로피들이 작아짐 -issue 그 이후에 변경한 값.

        for i in range(config.n_layer):
            # x = model.backbone['layers'][i].norm(residual)  
            # ====================  RMSNorm  ====================
            res2 = fxp32_16.mul(residual, residual)
            D = res2.shape[-1]
            res2_sum = res2.to(torch.int64).sum(dim=-1, keepdim=True) 
            mean_res2 = (res2_sum // D).clamp(fxp32_16.qmin, fxp32_16.qmax).to(torch.int32)
            eps_fxp = fxp32_16.quantize(torch.tensor(eps, dtype=torch.float32))
            mean_res2_eps = fxp32_16.add(mean_res2, eps_fxp)

            # 4. rsqrt 근사 (float으로 복원 후 근사 계산 → 다시 FXP16.13로 quant)
            mean_res2_eps_f = fxp32_16.dequantize(mean_res2_eps)
            rsqrt_approx_f = 1.0 / torch.sqrt(mean_res2_eps_f + 1e-6)
            rsqrt_out_fxp32 = fxp32_16.quantize(rsqrt_approx_f)  # FXP16.13

            rsqrt_broadcasted = rsqrt_out_fxp32.expand_as(residual)
            normed_fxp32 = fxp32_16.mul(residual, rsqrt_broadcasted)
            norm_weight_expanded = model.backbone['layers'][i].norm.weight.view(1, -1).expand_as(normed_fxp32)
            x_16 = fxp_multiply_and_cast_to_fxp16_11(normed_fxp32, norm_weight_expanded)
            # x = fxp16_11.dequantize(x_16)
            
            # residual * torch.rsqrt(res2.mean(-1, keepdim=True) + eps) * model.backbone['layers'][i].norm.weight
            # x = q_dq(x, 16, 11)  # -3.2 ~ 4.6, 11
            # x1_list[i].extend(x)
            # ==================================================

            # ====================  in_proj  ====================
            sc = quant_info[f"layers.{i}.mixer.in_proj.weight"]["scale"]
            zp = quant_info[f"layers.{i}.mixer.in_proj.weight"]["zero_point"]
            q_weight = model.backbone['layers'][i]['mixer'].in_proj.weight  # already FXP8.6
            x_input = x_16.squeeze(1)
            zxbcdt = fxp_linear_in_proj(x_input, q_weight, sc, zp, in_frac=11, w_frac=6, out_frac=11)


            # zxbcdt = model.backbone['layers'][i]['mixer'].in_proj(x.squeeze(1))
            z, xBC, dt = torch.split(
                zxbcdt,
                [config.d_inner, config.d_inner + 2 * config.d_state, config.nheads],
                dim=-1,
            )
            # xBC = q_dq(xBC, 16, 11)  # -6.1 ~ 6.1, 9
            # z = q_dq(z, 16, 11)  # -7.3 ~ 9
            # dt = q_dq(dt, 16, 11)  # -5.6 ~ 8.01
            # xBC1_list[i].extend(xBC)
            # z1_list[i].extend(z)
            # dt1_list[i].extend(dt)
            # ==================================================
            
            # ====================  conv1d  ====================
            h[i].conv_state.copy_(torch.roll(h[i].conv_state, shifts=-1, dims=-1))
            h[i].conv_state[:, :, -1] = xBC

            weight = rearrange(model.backbone['layers'][i]['mixer'].conv1d.weight, "d 1 w -> d w")
            bias = model.backbone['layers'][i]['mixer'].conv1d.bias

            xBC = fxp_conv1d_step(h[i].conv_state, weight, bias, in_frac=11, w_frac=6, bias_frac=4, out_frac=11)
# 
            # xBC = torch.sum(
            #     h[i].conv_state * rearrange(model.backbone['layers'][i]['mixer'].conv1d.weight, "d 1 w -> d w"),
            #     dim=-1
            # )
            # xBC += model.backbone['layers'][i]['mixer'].conv1d.bias
            # xBC = q_dq(xBC, 16, 11)  # -7.04 ~ 6
            # xBC2_list[i].extend(xBC)
            # ==================================================
            
            # ====================  SiLU  ====================
            xBC = silu_fxp_11to10(xBC)


            # xBC = F.silu(xBC)
            # xBC3_list[i].extend(xBC)
            # xBC = q_dq(xBC, 16, 10)  # -0.27 ~ 5.28, 얘는 11로 해도 되려나..
            
            x, B, C = torch.split(xBC, [config.d_inner, config.d_state, config.d_state], dim=-1)
            # x = q_dq(x, 16, 10)  # -0.27 ~ 5.28, 20, 얘도 작게 나올 수 있다.
            # B = q_dq(B, 16, 10) if (i == 19) else q_dq(B, 16, 12)  # -0.27 ~ 5.28
            # C = q_dq(C, 16, 11)  # -0.27 ~ 5.28, 17
            # x2_list[i].extend(x)
            # B_list[i].extend(B)
            # C_list[i].extend(C)
            # ==================================================
            
            # ====================  exp  ====================
            A = approx_exp(model.backbone['layers'][i]['mixer'].A_log)  # A_log: FXP16.13 float값 (실제로는 정수 기반)

            # A = -torch.exp(model.backbone['layers'][i]['mixer'].A_log)  # state decay factor
            # A = q_dq(A, 32, 16)  # -0.27 ~ 3.12 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # A_list[i].extend(A)
            # ==================================================

            # ====================  SoftPlus  ====================
            # 1. dt_bias 정렬 후 더함 (dt: FXP16.11 → FXP16.12로 정렬)
            dt_fixed = ((dt * 2) + model.backbone['layers'][i]['mixer'].dt_bias)  # FXP16.12

            # 2. softplus 근사 적용
            dt = approx_softplus(dt_fixed)  # 결과: FXP16.11

            # dt = F.softplus(dt + model.backbone['layers'][i]['mixer'].dt_bias)
            # dt = q_dq(dt, 16, 11)  # 0.00 ~ 8.8
            # dt2_list[i].extend(dt)
            # ==================================================

            # ====================  EWM  ====================
            dA = fxp_mul_dt_A(dt, A, dt_frac=11, A_frac=13, out_frac=16)  # dA: FXP32.16
            # dA = dt * A
            # dA = q_dq(dA, 32, 16)  # -18.6 ~ 0 !!!!!!!!!! layer4 이상
            # dA1_list[i].extend(dA)
            # ==================================================

            # ====================  exp  ====================
            dA = approx_exp_32(dA)
            # dA = torch.exp(dA)
            # dA = q_dq(dA, 16, 14)  # 0.00 ~ 1
            # dA2_list[i].extend(dA)
            # ==================================================
            
            # ====================  einsum  ====================
            x = rearrange(x, "b (h p) -> b h p", p=config.headdim)

            # dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
            dBx = fxp_mul3_dtBx(dt, B, x, dt_frac=11, B_frac=10, x_frac=10, out_frac=11)
            # dBX = q_dq(dBx, 16, 11)  # -2.1 ~ 2.9, 9  2.7b에서는 -1 ~ 1 이 나옴
            # dBx_list[i].extend(dBx)
            # ==================================================
            
            # ====================  EWM  ====================
            # dAh = h[i].ssm_state * rearrange(dA, "b h -> b h 1 1")
            dAh = fxp_mul_dAh(h[i].ssm_state, dA, h_frac=10, dA_frac=14, out_frac=10)
            # dAh = q_dq(dAh, 16, 10)  # -6.0636 ~ 10.6496, 22
            # dAh_list[i].extend(dAh)
            # ==================================================
            
            # ====================  EWA  ====================
            dAhdBx = fxp_add_dAh_dBx(dAh, dBx)  # 출력: float32 값이지만 의미는 FXP16.10
            # dAhdBx = dAh + dBx
            # dAhdBx = q_dq(dAhdBx, 16, 10)  # -6.585 ~ 12.080, 22
            # dAhdBx_list[i].extend(dAhdBx)
            # ==================================================
            
            # ====================  MM  ====================
            h[i].ssm_state.copy_(dAhdBx)
            y = fxp_dot_ssm_C(h[i].ssm_state, C, ssm_frac=10, C_frac=10, out_frac=7)
            # y = torch.einsum("bhpn, bn -> bhp", h[i].ssm_state, C)
            # y = q_dq(y, 16, 7)  # -18.287 ~ 56.663, 210
            # y1_list[i].extend(y)
            # ==================================================
            
            # ====================  add  ====================
            # y = y + rearrange(model.backbone['layers'][i]['mixer'].D, "h -> h 1") * x
            y = fxp_add_Dx_to_y(y, x, model.backbone['layers'][i]['mixer'].D, x_frac=10, D_frac=5, y_frac=7)
            y = rearrange(y, "b h p -> b (h p)")
            # y = q_dq(y, 16, 7)  # -17.542 ~ 72.729, 250
            # y2_list[i].extend(y)
            # ==================================================
            
            # ====================  SiLU  ====================
            z = silu_fxp_11to11(z)
            # z = F.silu(z)
            # z = q_dq(z, 16, 11)# -0.278 ~ 6.931
            # z2_list[i].extend(z)
            
            # ====================  MM  ====================
            y = fxp_mul_y_z_to_32_16(y, z)  # y: FXP32.16 의미
            # y = y * z
            # y = q_dq(y, 32, 16) # -29.336 ~ 212.738, 600  # 이건 더 커질수도
            # y3_list[i].extend(y)
            # ==================================================

            # ====================  RMSNorm  ====================
            # 1. y^2 계산
            y2 = fxp32_16.mul(y, y)
            D = y2.shape[-1]

            # 2. 평균 및 epsilon 더하기
            y2_sum = y2.to(torch.int64).sum(dim=-1, keepdim=True) 
            mean_y2 = (y2_sum // D).clamp(fxp32_16.qmin, fxp32_16.qmax).to(torch.int32)
            mean_y2_eps = fxp32_16.add(mean_y2, eps_fxp)

            # 3. 역제곱근 근사 (dequant → float domain 계산 → re-quant)
            mean_y2_eps_f = fxp32_16.dequantize(mean_y2_eps)
            rsqrt_approx_f_y = 1.0 / torch.sqrt(mean_y2_eps_f + 1e-6)
            rsqrt_out_fxp32_y = fxp32_16.quantize(rsqrt_approx_f_y)  # FXP16.13

            # 4. 정규화
            rsqrt_broadcasted = rsqrt_out_fxp32_y.expand_as(y)
            normed_fxp32_y = fxp32_16.mul(y, rsqrt_broadcasted)

            # 5. norm.weight 처리
            norm_weight = model.backbone['layers'][i].norm.weight  # shape: [2560]

            # 👉 중간 feature 확장 대응
            if normed_fxp32_y.shape[1] == 2 * norm_weight.shape[0]:
                norm_weight = norm_weight.repeat(2)  # shape: [5120]
            elif normed_fxp32_y.shape[1] != norm_weight.shape[0]:
                raise ValueError(
                    f"Mismatch: norm.weight={norm_weight.shape}, normed_y={normed_fxp32_y.shape}"
                )

            # 6. weight broadcast 및 scaling 적용
            norm_weight_expanded_y = norm_weight.view(1, -1).expand_as(normed_fxp32_y)
            y = fxp_multiply_and_cast_to_fxp16_11(normed_fxp32_y, norm_weight_expanded_y, out_bit=32, out_frac=16)

            # y = model.backbone['layers'][i]['mixer'].norm(y)
            # y = q_dq(y, 32, 16)
            # y4_list[i].extend(y)
            # ==================================================
            
            # ====================  out_proj  ====================
            sc = quant_info[f"layers.{i}.mixer.out_proj.weight"]["scale"]
            zp = quant_info[f"layers.{i}.mixer.out_proj.weight"]["zero_point"]
            q_weight = model.backbone['layers'][i]['mixer'].out_proj.weight  # already FXP8.6
            y_input = y.squeeze(1)
            y = fxp_linear_out_proj_32in_32out(y_input, q_weight, sc, zp, in_frac=16, w_frac=6, out_frac=16)

            # y = model.backbone['layers'][i]['mixer'].out_proj(y)
            # y = q_dq(y, 32, 16)
            # y5_list[i].extend(y)
            # ==================================================
            
            # ====================  residual  ====================
            residual = residual + y.unsqueeze(1)
            # residual_list[i].extend(y)
            # ==================================================

        residual = fxp32_16.dequantize(residual)
        residual = model.backbone.norm_f(residual)
        logits = model.lm_head(residual)  # shape: (1, 1, vocab_size)
        out = logits[:, :seqlen]  # seqlen=1
        logits = out[0, -1]  # 최종 토큰의 로짓
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 종료 조건 확인 (EOS 토큰일 경우)
        if next_token.item() == tokenizer.eos_token_id:
            break

        # 다음 루프 준비
        generated.append(next_token.item())
        # f_token = next_token.unsqueeze(0)
        f_token = next_token.to(device).unsqueeze(0)

# 토큰 결과 디코딩 후 출력
print(tokenizer.decode(generated, skip_special_tokens=True))


'''
# list_l('x1', x1_list)
# list_l('xBC1', xBC1_list)
# list_l('z1', z1_list)
# list_l('dt1', dt1_list)
# list_l('xBC2', xBC2_list)
# list_l('xBC3', xBC3_list)
# list_l('x2', x2_list)
# list_l('B', B_list)
# list_l('C', C_list)
# list_l('A', A_list)
# list_l('dt2', dt2_list)
# list_l('dA1', dA1_list)
# list_l('dA2', dA2_list)
# list_l('dBx', dBx_list)
# list_l('dAh', dAh_list)
# list_l('dAhdBx', dAhdBx_list)
# list_l('y1', y1_list)
# list_l('y2', y2_list)
# list_l('z2', z2_list)
# list_l('y3', y3_list)
# list_l('y4', y4_list)
# list_l('y5', y5_list)
# list_l('residual', residual)
'''


# logger.info(f"  Entropy: \n{entrophy_list}\n")
# logger.info(f"  Cross Entropy Loss: \n{loss_list}\n")

'''            
# import matplotlib.pyplot as plt

# steps = list(range(len(entrophy_list)))

# plt.plot(steps, entrophy_list, label='Entropy', marker='o')
# plt.plot(steps, loss_list, label='Cross Entropy Loss', marker='x')

# plt.xlabel("Step (Token Index)")
# plt.ylabel("Entropy / Loss")
# plt.title("Entropy and Cross Entropy Loss per Step")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
'''