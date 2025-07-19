'''
    mamba2-130m quantization FXP8/16 standard
'''

# 필요한 라이브러리 import
import time
import torch
from torch import nn
from transformers import AutoTokenizer
from mamba2 import Mamba2LMHeadModel, Mamba2Config, ssd, InferenceCache
import torch.nn.functional as F
from einops import rearrange

# CUDA가 가능하면 GPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mamba2 설정 정의: d_model=768, 24-layer, vocab_size=50277
config = Mamba2Config(d_model=768, n_layer=24, vocab_size=50277)

# 모델 초기화 및 양자화된 state_dict 로딩
model = Mamba2LMHeadModel(config)
model.load_state_dict(torch.load(r"C:\Internship\mamba2-130m\mamba2_130m_quantized.pth"))
model = model.to(device)
# for name, param in model.named_parameters():
#     print(f"{name}: shape = {param.shape}, requires_grad = {param.requires_grad}")

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
prompt = "The future of AI"
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
    for t in range(100):
        seqlen = f_token.shape[1]  # 보통 1

        # f_token을 모델 입력 형식으로 변환
        # input_tensor = torch.tensor([[f_token]], device=device)
        input_tensor = f_token.to(device)

        # (1, 1) → 임베딩: (1, 1, d_model)
        u = model.backbone['embedding'](input_tensor)
        residual = u  # skip connection용

        # 전체 레이어 순차 처리
        for i in range(config.n_layer):
            x = model.backbone['layers'][i].norm(residual)  # RMSNorm

            # Linear projection → z, xBC, dt 분리
            zxbcdt = model.backbone['layers'][i]['mixer'].in_proj(x.squeeze(1))
            z, xBC, dt = torch.split(
                zxbcdt,
                [config.d_inner, config.d_inner + 2 * config.d_state, config.nheads],
                dim=-1,
            )

            # convolution을 위한 상태 이동 및 입력 추가
            h[i].conv_state.copy_(torch.roll(h[i].conv_state, shifts=-1, dims=-1))
            h[i].conv_state[:, :, -1] = xBC

            # 1D depthwise convolution (수동 구현)
            xBC = torch.sum(
                h[i].conv_state * rearrange(model.backbone['layers'][i]['mixer'].conv1d.weight, "d 1 w -> d w"),
                dim=-1
            )
            xBC += model.backbone['layers'][i]['mixer'].conv1d.bias
            xBC = F.silu(xBC)

            # x, B, C 분리
            x, B, C = torch.split(xBC, [config.d_inner, config.d_state, config.d_state], dim=-1)

            A = -torch.exp(model.backbone['layers'][i]['mixer'].A_log)  # state decay factor

            # SSM 계산
            dt = F.softplus(dt + model.backbone['layers'][i]['mixer'].dt_bias)
            dA = torch.exp(dt * A)
            x = rearrange(x, "b (h p) -> b h p", p=config.headdim)

            # 상태 업데이트
            dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
            h[i].ssm_state.copy_(h[i].ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)

            # 출력 계산
            y = torch.einsum("bhpn, bn -> bhp", h[i].ssm_state, C)
            y = y + rearrange(model.backbone['layers'][i]['mixer'].D, "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            y = model.backbone['layers'][i]['mixer'].norm(y, z)
            y = model.backbone['layers'][i]['mixer'].out_proj(y)

            # residual connection
            residual = residual + y.unsqueeze(1)

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
