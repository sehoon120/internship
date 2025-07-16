import time
import torch
from torch import nn
from transformers import AutoTokenizer
from mamba2 import Mamba2LMHeadModel, Mamba2Config, ssd, InferenceCache
import torch.nn.functional as F
from einops import rearrange

# 내부 구조 분석 진행중 코드
# 정상 동작 확인

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Mamba2Config(
    d_model=768, 
    n_layer=24, 
    vocab_size=50277
    )  # 130m
# config = Mamba2Config(d_model=256, n_layer=6, vocab_size=50277)
model = Mamba2LMHeadModel(config)
model.load_state_dict(torch.load(r"C:\Internship\mamba2-130m\mamba2_130m_quantized.pth"))
# print(model)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id
h = [InferenceCache.alloc(
    batch_size=1,  # 배치 크기
    args=config,
    device=device
) for _ in range(config.n_layer)]
# print("conv_state:", h.conv_state.shape)  # (1, 1792, 4)
# print("ssm_state:", h.ssm_state.shape)    # (1, 24, 64, 128)

# 입력 프롬프트
prompt = "The future of AI"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)  # (1, L)
prefix, f_token = input_ids[:, :-1], input_ids[:, -1:]  # prefix: "The future of"  f_token: "AI"

chunk_size = model.args.chunk_size  # ex: 64
n_chunked = (prefix.shape[1] // chunk_size) * chunk_size
# print('prefix.shape[1]: ', prefix.shape[1], ', n_chunked: ', n_chunked)
if n_chunked > 0:
    _, h = model(prefix[:, :n_chunked], None)    # forward로 처리 h 채우기
else:
    h = [InferenceCache.alloc(1, model.args, device=device) for _ in range(model.args.n_layer)]

for i in range(n_chunked, prefix.shape[1]):
    _, h = model(prefix[:, i:i+1], h)  # 남은 프롬프트 step으로 수행


# tokens = input_ids[0].tolist()  # ex: [502, 321, 764]


generated = [t.item() for t in input_ids[0]]  # 결과 누적 list
with torch.no_grad():
    for _ in range(20):
        seqlen = f_token.shape[1]
        input_tensor = torch.tensor([[f_token]], device=device)

        # input_ids: (batch_size, seq_len) → torch.LongTensor
        u = model.backbone['embedding'](input_tensor)  # u: (batch_size, seq_len, d_model)
        residual = u
        for i in range(config.n_layer):  # model(tokens, h)
            x = model.backbone['layers'][i].norm(residual)
            # 1. projection
            assert x.shape[1] == 1
            zxbcdt = model.backbone['layers'][i]['mixer'].in_proj(x.squeeze(1))  # shape: (B, L, D_in_proj)
            z, xBC, dt = torch.split(
                zxbcdt,
                [config.d_inner, config.d_inner + 2 * config.d_state, config.nheads],
                dim=-1,
            )
            # print('z: ', z.shape)
            # print('xBC: ', xBC.shape)
            # print('dt: ', dt.shape)

            h[i].conv_state.copy_(torch.roll(h[i].conv_state, shifts=-1, dims=-1))
            h[i].conv_state[:, :, -1] = xBC
            # print('xBC: ', xBC.shape)
            xBC = torch.sum(
                h[i].conv_state * rearrange(model.backbone['layers'][i]['mixer'].conv1d.weight, "d 1 w -> d w"), dim=-1
            )
            # print('h[i].conv_state3', h[i].conv_state.shape)
            # print('xBC2: ', xBC.shape)
            xBC += model.backbone['layers'][i]['mixer'].conv1d.bias
            xBC = F.silu(xBC)
            x, B, C = torch.split(xBC, [config.d_inner, config.d_state, config.d_state], dim=-1)
            # print('x: ', x.shape)
            # print('B: ', B.shape)
            # print('C: ', C.shape)
            A = -torch.exp(model.backbone['layers'][i]['mixer'].A_log)  # (nheads,)
            # print('A: ', A.shape)

            # SSM step
            dt = F.softplus(dt + model.backbone['layers'][i]['mixer'].dt_bias)  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=config.headdim)  # "b l (h p) -> b l h p"
            # print('dt: ', dt.shape)
            # print('B: ', B.shape)
            # print('X: ', x.shape)
            dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
            h[i].ssm_state.copy_(h[i].ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn, bn -> bhp", h[i].ssm_state, C)
            y = y + rearrange(model.backbone['layers'][i]['mixer'].D, "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            y = model.backbone['layers'][i]['mixer'].norm(y, z)
            y = model.backbone['layers'][i]['mixer'].out_proj(y)

            residual = residual + y.unsqueeze(1)
            # return y.unsqueeze(1), h[i]
            # print(y.unsqueeze(1)[0,0,:5])

        residual = model.backbone.norm_f(residual)
        logits = model.lm_head(residual)  # LMHead
        out =  logits[:, :seqlen]
        logits = out[0,-1]

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated.append(next_token.item())
        f_token = next_token.unsqueeze(0)

print(tokenizer.decode(generated, skip_special_tokens=True))