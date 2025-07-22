'''
    mamba2-130m quantization FXP8/16 standard

    엔트로피가 작게나오는 issue가 있음
    residual쪽은 32bit으로 하는 방법 고려해보기
'''
def findm(x):
    max_val = x.max()     # 최대값
    min_val = x.min()     # 최소값
    mean_val = x.mean()   # 평균값
    print(f"Max: {max_val}, Min: {min_val}, Mean: {mean_val}")
    return 0

# 필요한 라이브러리 import
import time
import torch
from torch import nn
from transformers import AutoTokenizer
from mamba2 import Mamba2LMHeadModel, Mamba2Config, ssd, InferenceCache
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

config = Mamba2Config(d_model=768, n_layer=24, vocab_size=50277)
# config = Mamba2Config(d_model=2560, n_layer=64, vocab_size=50288)

# model = Mamba2LMHeadModel(config)
# model.load_state_dict(torch.load(r"C:\Internship\mamba2-130m\mamba2_130m_quantized.pth"))
# # model.load_state_dict(torch.load(r"C:\Internship\mamba2-2.7b\mamba2_2.7b_quantized.pth"))
# model = model.to(device)

model_name = 'state-spaces/mamba2-130m'  # "AntonV/mamba2-130m-hf"
model = Mamba2LMHeadModel.from_pretrained(model_name, device=device)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id
h = [InferenceCache.alloc(
    batch_size=1,
    args=config,
    device=device
) for _ in range(config.n_layer)]

prompt = """
Mamba is a new sequence model that can replace transformers in some cases. 
It uses state space models instead of attention. Its advantage is that it is faster and more memory-efficient.

Write a clear summary of how Mamba differs from Transformers.
"""
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)  # shape: (1, L)

prefix, f_token = input_ids[:, :-1], input_ids[:, -1:]
chunk_size = model.args.chunk_size
n_chunked = (prefix.shape[1] // chunk_size) * chunk_size
if n_chunked > 0:
    _, h = model(prefix[:, :n_chunked], None)
else:
    h = [InferenceCache.alloc(1, model.args, device=device) for _ in range(model.args.n_layer)]
for i in range(n_chunked, prefix.shape[1]):
    _, h = model(prefix[:, i:i+1], h)

generated = [t.item() for t in input_ids[0]]

entrophy_list = []
rms1_list = []

with torch.no_grad():
    for t in range(50):  # 100
        seqlen = f_token.shape[1]
        input_tensor = f_token.to(device)
        u = model.backbone['embedding'](input_tensor)
        residual = u  
        residual = q_dq(residual, 16, 10)  # -2.593 ~ 3.061
        # 이거 하니까 엔트로피들이 작아짐 -issue 그 이후에 변경한 값.
        # rms1_list.append(residual) 

        for i in range(config.n_layer):
            x = model.backbone['layers'][i].norm(residual)  # RMSNorm
            x = q_dq(x, 16, 11)  # -3.2 ~ 4.3
            
            zxbcdt = model.backbone['layers'][i]['mixer'].in_proj(x.squeeze(1))
            z, xBC, dt = torch.split(
                zxbcdt,
                [config.d_inner, config.d_inner + 2 * config.d_state, config.nheads],
                dim=-1,
            )
            xBC = q_dq(xBC, 8, 4)  # -6.1 ~ 6.1
            z = q_dq(z, 8, 4)  # -7.3 ~ 6.6
            dt = q_dq(dt, 8, 4)  # -5.6 ~ 8.01
            
            h[i].conv_state.copy_(torch.roll(h[i].conv_state, shifts=-1, dims=-1))
            h[i].conv_state[:, :, -1] = xBC

            xBC = torch.sum(
                h[i].conv_state * rearrange(model.backbone['layers'][i]['mixer'].conv1d.weight, "d 1 w -> d w"),
                dim=-1
            )
            xBC += model.backbone['layers'][i]['mixer'].conv1d.bias
            xBC = q_dq(xBC, 8, 4)  # -7.04 ~ 5.4
            
            xBC = F.silu(xBC)
            xBC = q_dq(xBC, 16, 12)  # -0.27 ~ 5.28
            
            x, B, C = torch.split(xBC, [config.d_inner, config.d_state, config.d_state], dim=-1)
            x = q_dq(x, 16, 12)  # -0.27 ~ 5.28
            B = q_dq(B, 16, 12)  # -0.27 ~ 5.28
            C = q_dq(C, 16, 12)  # -0.27 ~ 5.28
            
            A = -torch.exp(model.backbone['layers'][i]['mixer'].A_log)  # state decay factor
            A = q_dq(A, 8, 5)  # -0.27 ~ 3.12
            
            dt = F.softplus(dt + model.backbone['layers'][i]['mixer'].dt_bias)
            dt = q_dq(dt, 16, 11)  # 0.00 ~ 8.8
            
            dA = dt * A
            dA = q_dq(dA, 16, 10)  # -18.6 ~ 0
                        
            dA = torch.exp(dA)
            dA = q_dq(dA, 16, 14)  # 0.00 ~ 1
            
            x = rearrange(x, "b (h p) -> b h p", p=config.headdim)

            dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
            dBX = q_dq(dBx, 16, 13)  # -2.1 ~ 2.9
            
            dAh = h[i].ssm_state * rearrange(dA, "b h -> b h 1 1")
            dAh = q_dq(dAh, 16, 11)  # -6.0636 ~ 10.6496
            

            dAhdBx = dAh + dBx
            dAhdBx = q_dq(dAhdBx, 16, 11)  # -6.585 ~ 12.080 여기부터 좀 이상해짐
            
            h[i].ssm_state.copy_(dAhdBx)
            y = torch.einsum("bhpn, bn -> bhp", h[i].ssm_state, C)
            y = q_dq(y, 16, 9)  # -18.287 ~ 56.663
            
            y = y + rearrange(model.backbone['layers'][i]['mixer'].D, "h -> h 1") * x
            y = q_dq(y, 16, 8)  # -17.542 ~ 72.729
            
            y = rearrange(y, "b h p -> b (h p)")
            z = F.silu(z)
            z = q_dq(z, 16, 12)# -0.278 ~ 6.931
            
            y = y * z
            y = q_dq(y, 16, 7)  # -29.336 ~ 212.738  # 이건 더 커질수도

            y = model.backbone['layers'][i]['mixer'].norm(y)
            y = q_dq(y, 16, 9)  # -34.582 ~ 30.436
            
            y = model.backbone['layers'][i]['mixer'].out_proj(y)
            y = q_dq(y, 16, 7)  # -134.434 ~ 143.244  # 이거 하니까 엔트로피들이 작아짐 -issue
            
            residual = residual + y.unsqueeze(1)

        residual = model.backbone.norm_f(residual)
        logits = model.lm_head(residual)  # shape: (1, 1, vocab_size)
        out = logits[:, :seqlen]  # seqlen=1
        logits = out[0, -1]  # 최종 토큰의 로짓
        probs = F.softmax(logits, dim=-1)

        entrophy_list.append(-(probs * probs.log()).sum(dim=-1).mean())

        # print("Output entropy:", entropy.item())

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

# rms1_list_cat = torch.cat([x.flatten() for x in rms1_list])
# min_val = np.percentile(rms1_list_cat, 0.05)   # 하위 0.05%
# max_val = np.percentile(rms1_list_cat, 99.95)  # 상위 0.05%
# print(f"\n==================================================\nglobal min/max: {min_val:.3f} ~ {max_val:.3f}\n==================================================\n")

import matplotlib.pyplot as plt

steps = list(range(len(entrophy_list)))

plt.plot(steps, entrophy_list, label='Float Model', marker='o')
# plt.plot(steps, quant_entropy_list, label='Quantized Model', marker='x')
plt.xlabel("Step (token index)")
plt.ylabel("Entropy")
plt.title("Entropy per Step")
plt.legend()
plt.grid(True)
plt.show()