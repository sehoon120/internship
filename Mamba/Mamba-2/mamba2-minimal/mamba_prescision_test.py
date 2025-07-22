# 필요한 라이브러리 import
import time
import torch
from torch import nn
from transformers import AutoTokenizer
from mamba2 import Mamba2LMHeadModel, Mamba2Config, ssd, InferenceCache
import torch.nn.functional as F
from einops import rearrange

def analyze_fxp_precision(tensor, max_total_bits=16, target_error=1e-2):
    import torch
    import numpy as np

    min_val = tensor.min().item()
    max_val = tensor.max().item()
    range_val = max_val - min_val

    print(f"Range: {min_val:.4f} to {max_val:.4f}")

    best_config = None
    lowest_error = float('inf')

    for total_bits in range(8, max_total_bits + 1):
        for frac_bits in range(1, total_bits):
            int_bits = total_bits - frac_bits
            scale = 2 ** frac_bits
            q = torch.round(tensor * scale).clamp(-(2**(int_bits-1)), 2**(int_bits-1) - 1)
            dq = q / scale

            error = torch.abs(tensor - dq).mean().item()
            if error < lowest_error:
                lowest_error = error
                best_config = (total_bits, frac_bits, error)

            if error < target_error:
                return {"total_bits": total_bits, "frac_bits": frac_bits, "abs_error": error}

    return {
        "total_bits": best_config[0],
        "frac_bits": best_config[1],
        "abs_error": best_config[2],
        "note": "target error not reached, best effort"
    }


from FXP_simulator import FXP16Simulator, FXP32Simulator, FXP8Simulator
fxp16_14 = FXP16Simulator(frac_bits=14)
fxp16_12 = FXP16Simulator(frac_bits=12)
fxp16_11 = FXP16Simulator(frac_bits=11)
fxp16_10 = FXP16Simulator(frac_bits=10)
fxp16_8 =  FXP16Simulator(frac_bits=8)
fxp16_5 =  FXP16Simulator(frac_bits=5)
fxp16_4 =  FXP16Simulator(frac_bits=4)
fxp32_16 = FXP32Simulator(frac_bits=16)
fxp32_21 = FXP32Simulator(frac_bits=21)
fxp8_6 = FXP8Simulator(frac_bits=6)
fxp8_4 = FXP8Simulator(frac_bits=4)

from exp_graph import FastBiasedExp

from RMS_graph import InvSqrtApprox16Segment

from SiLU_graph import ApproxSiLU16
approx = ApproxSiLU16()

def findm(x):
    max_val = x.max()     # 최대값
    min_val = x.min()     # 최소값
    mean_val = x.mean()   # 평균값
    print(f"Max: {max_val}, Min: {min_val}, Mean: {mean_val}")
    return 0

def q_dq(x, a, b):
    if a == 16:
        if b == 4:
            return fxp16_4.dequantize(fxp16_4.quantize(x))
        elif b == 5:
            return fxp16_5.dequantize(fxp16_5.quantize(x))
        elif b == 8:
            return fxp16_8.dequantize(fxp16_8.quantize(x))
        elif b == 10:
            return fxp16_10.dequantize(fxp16_10.quantize(x))
        elif b == 11:
            return fxp16_11.dequantize(fxp16_11.quantize(x))
        elif b == 12:
            return fxp16_12.dequantize(fxp16_12.quantize(x))
        elif b == 14:
            return fxp16_14.dequantize(fxp16_14.quantize(x))
    elif a == 32:
        if b == 16:
            return fxp32_16.dequantize(fxp32_16.quantize(x))
        if b == 21:
            return fxp32_21.dequantize(fxp32_21.quantize(x))
    elif a == 8:
        if b == 4:
            return fxp8_4.dequantize(fxp8_4.quantize(x))
        elif b == 6:
            return fxp8_6.dequantize(fxp8_6.quantize(x))


# CUDA가 가능하면 GPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mamba2 설정 정의: d_model=768, 24-layer, vocab_size=50277
# config = Mamba2Config(d_model=768, n_layer=24, vocab_size=50277)
config = Mamba2Config(d_model=2560, n_layer=64, vocab_size=50288)

# 모델 초기화 및 양자화된 state_dict 로딩
model = Mamba2LMHeadModel(config)
# model.load_state_dict(torch.load(r"C:\Internship\mamba2-130m\mamba2_130m_quantized.pth"))
model.load_state_dict(torch.load(r"C:\Internship\mamba2-2.7b\mamba2_2.7b_quantized.pth"))
model = model.to(device)

# GPT-NeoX 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id

# 레이어 수만큼 inference cache (conv_state, ssm_state) 초기화
h = [InferenceCache.alloc(
    batch_size=1,
    args=config,
    device=device
) for _ in range(config.n_layer)]

# 입력 프롬프트 정의 및 토크나이징
# prompt = "Mamba is a new sequence model that can replace transformers in some cases. It uses state space models instead of attention. Its advantage is that it is faster and more memory-efficient."  # "The future of AI"
prompt = """
I'm happy.
How about you?
"""
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)  # shape: (1, L)

# 마지막 토큰을 f_token으로 분리, 나머지는 prefix
prefix, f_token = input_ids[:, :-1], input_ids[:, -1:]

# 모델 내부 chunk size를 기준으로 처리 가능한 구간 계산
chunk_size = model.args.chunk_size
n_chunked = (prefix.shape[1] // chunk_size) * chunk_size

# chunk 단위는 한번에 처리하고, 나머지는 step-by-step으로 처리
if n_chunked > 0:
    _, h = model(prefix[:, :n_chunked], None)
else:
    h = [InferenceCache.alloc(1, model.args, device=device) for _ in range(model.args.n_layer)]

# 남은 prefix 토큰들 처리 (inference step 방식)
for i in range(n_chunked, prefix.shape[1]):
    _, h = model(prefix[:, i:i+1], h)


# 생성된 결과 토큰 저장 리스트 초기화
generated = [t.item() for t in input_ids[0]]
# 자동 생성 반복: 최대 20 토큰 생성
with torch.no_grad():
    # for _ in range(1):  # 20
    for _ in range(50):  # 100
        seqlen = f_token.shape[1]  # 보통 1

        # f_token을 모델 입력 형식으로 변환
        # input_tensor = torch.tensor([[f_token]], device=device)
        input_tensor = f_token.to(device)

        # (1, 1) → 임베딩: (1, 1, d_model)
        u = model.backbone['embedding'](input_tensor)

        # print(u)
        # u_p = u
        # findm(u) # -8 ~ 5
        # u = q_dq(u, 16, 11)  # 0.08 오차 발생  # fxp16_12 -> 0.02 but outliers
        
        # print("Abs Error:  ", torch.sum(torch.abs(u - u_p)))

        residual = u  # skip connection용
        residual = q_dq(residual, 32, 16)

        # 전체 레이어 순차 처리
        # for i in range(1):  # config.n_layer
        for i in range(config.n_layer):
            # residual = torch([[[0, 00.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]]])
            x = model.backbone['layers'][i].norm(residual)  # RMSNorm
            # findm(x)  # -9 ~ 9
            # x_p = x
            x = q_dq(x, 16, 11)  # -3 ~ 4
            # print(f"|{i}| Abs Error:  ", torch.sum(torch.abs(x - x_p)))  # 0.09
# '''
            # Linear projection → z, xBC, dt 분리
            zxbcdt = model.backbone['layers'][i]['mixer'].in_proj(x.squeeze(1))
            z, xBC, dt = torch.split(
                zxbcdt,
                [config.d_inner, config.d_inner + 2 * config.d_state, config.nheads],
                dim=-1,
            )
            # findm(xBC)
            xBC = q_dq(xBC, 16, 11)  # 13 ~ -15  outlier -19

            # xBC_p = xBC
            # xBC = fxp16_11.quantize(xBC)
            # xBC = fxp16_11.dequantize(xBC)
            # print(f"y {i} Abs Error:", torch.sum(torch.abs(xBC - xBC_p)))



            # convolution을 위한 상태 이동 및 입력 추가
            h[i].conv_state.copy_(torch.roll(h[i].conv_state, shifts=-1, dims=-1))
            h[i].conv_state[:, :, -1] = xBC

            # 1D depthwise convolution (수동 구현)
            xBC = torch.sum(
                h[i].conv_state * rearrange(model.backbone['layers'][i]['mixer'].conv1d.weight, "d 1 w -> d w"),
                dim=-1
            )

            xBC += model.backbone['layers'][i]['mixer'].conv1d.bias

            xBC_q = q_dq(xBC, 16, 10)  # 10, 11 둘 다 평균 0.02로 비슷
            # xBC = F.silu(xBC)
            xBC = approx(xBC_q)  # 근사 SiLU 사용
            # print(f"|{i}| {a.shape} Abs Error:", torch.sum(torch.abs(a - xBC)))
            


            # x, B, C 분리
            x, B, C = torch.split(xBC, [config.d_inner, config.d_state, config.d_state], dim=-1)
            x = q_dq(x, 16, 10)  # x: -21 ~ 1
            B = q_dq(B, 16, 10)  # B: +-4 but outlier 20
            C = q_dq(C, 16, 12)  # C: -3 ~ 0 outlier -6

            A = -torch.exp(model.backbone['layers'][i]['mixer'].A_log)  # state decay factor
            # A = -FastBiasedExp()(model.backbone['layers'][i]['mixer'].A_log)

            # SSM 계산
            
            dt = q_dq(dt, 16, 12)  # 0.0002 softplus 통과 과정에서
            dt = F.softplus(dt + model.backbone['layers'][i]['mixer'].dt_bias)  # 10 ~ 0
            dt = q_dq(dt, 16, 11)

            dA = torch.exp(dt * A)  # 1 ~ 0
            dA = q_dq(dA, 16, 14)
            # dA = FastBiasedExp()(dt * A)

# SSM start

            x = rearrange(x, "b (h p) -> b h p", p=config.headdim)

            # 상태 업데이트
            dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
            h[i].ssm_state.copy_(h[i].ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)

            # 출력 계산
            y = torch.einsum("bhpn, bn -> bhp", h[i].ssm_state, C)
            y = y + rearrange(model.backbone['layers'][i]['mixer'].D, "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")

# SSM end
            
            y = q_dq(y, 16, 5)  # +-420  오차 0.007
            

            z = q_dq(z, 32, 16)  # 16 ~ -10
            z = approx(z)  # 16 ~ -10
            z = q_dq(z, 16, 11)
            
            y = y * z
            
            # result = analyze_fxp_precision(y)
            # print(result)
            
            y = q_dq(y, 32, 16)  # error 0
            # y = q_dq(y, 16, 4)  # errror 0.014
            

            y = model.backbone['layers'][i]['mixer'].norm(y)

            # a = y
            y = q_dq(y, 32, 16)  # error 0.000004
            # print(f"|{i}| {a.shape} Abs Error:", torch.sum(torch.abs(y - a)))

            y = model.backbone['layers'][i]['mixer'].out_proj(y)  # +-450
            y = q_dq(y, 32, 16)
            # residual connection
            residual = residual + y.unsqueeze(1)



            # max_val = residual.max().item()
            # min_val = residual.min().item()

            # print(f"{i}  최댓값:", max_val, "최솟값:", min_val)

            
            residual_p = residual
            # # residual_q = fxp32.quantize(residual_p)
            # # residual_dq = fxp32.dequantize(residual_q)
            # residual = residual_dq

        # 최종 layer norm 후 LM 헤드
        residual = model.backbone.norm_f(residual)
        logits = model.lm_head(residual)  # shape: (1, 1, vocab_size)
        out = logits[:, :seqlen]  # seqlen=1
        logits = out[0, -1]  # 최종 토큰의 로짓

        # 다음 토큰 샘플링
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
# '''