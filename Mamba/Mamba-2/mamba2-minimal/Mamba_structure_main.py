"""
    이 코드가 제작할 모델 구조의 확정 버전이 될것이다.

    현재 제작중

    SSM Block의 구현에 집중
    FP16 operation in SSM Block

    FXP8 with several components
"""
import time
import torch
from torch import nn
from transformers import AutoTokenizer
from mamba2 import Mamba2LMHeadModel, Mamba2Config, ssd, InferenceCache
import torch.nn.functional as F
from einops import rearrange
import os

# 이 부분 고치던중
current_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(current_dir, "../../../..", "intermediate_datas")
os.makedirs(save_dir, exist_ok=True)  # 디렉토리 없으면 생성

# FP16 텐서를 .hex로 저장하는 함수
def save_tensor_fp16_hex(tensor: torch.Tensor, filename: str):
    tensor_fp16 = tensor.to(torch.float16).flatten()
    u16_tensor = tensor_fp16.view(torch.uint16).cpu().numpy()  # vectorized

    with open(filename, 'w') as f:
        for val in u16_tensor:
            f.write(f"{val:04x}\n")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Mamba2Config(
    d_model=768, 
    n_layer=24, 
    vocab_size=50277
    )  # 130m
# model = Mamba2LMHeadModel(config)
# model.load_state_dict(torch.load(r"C:\Internship\mamba2-130m\mamba2_130m_quantized.pth"))

model_name = 'state-spaces/mamba2-130m'
model = Mamba2LMHeadModel.from_pretrained(model_name, device=device)

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
prompt = """
Continue the story: "The robot slowly opened the door, not knowing what it would find on the other side..."
"""
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

average_list = []
average_list_m = []
generated = [t.item() for t in input_ids[0]]  # 결과 누적 list
with torch.no_grad():
    for token_num in range(50):
        t1_m = time.perf_counter()
        ssm_time = []
        mamba_time = []

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

            h[i].conv_state.copy_(torch.roll(h[i].conv_state, shifts=-1, dims=-1))
            h[i].conv_state[:, :, -1] = xBC

            xBC = torch.sum(
                h[i].conv_state * rearrange(model.backbone['layers'][i]['mixer'].conv1d.weight, "d 1 w -> d w"), dim=-1
            )

            xBC += model.backbone['layers'][i]['mixer'].conv1d.bias
            xBC = F.silu(xBC)
            x, B, C = torch.split(xBC, [config.d_inner, config.d_state, config.d_state], dim=-1)

            A = -torch.exp(model.backbone['layers'][i]['mixer'].A_log)  # (nheads,)

            # SSM step
            dt = F.softplus(dt + model.backbone['layers'][i]['mixer'].dt_bias)  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=config.headdim)  # "b l (h p) -> b l h p"
# ====================  SSM Block Started  ====================
# input: dA, dt, x, B, C, D, h[i].ssm_state  -  FP16
# output: y  -  FP16
            # FP32 -> FP16
            dA = dA.to(dtype=torch.float16)
            dt = dt.to(dtype=torch.float16)
            x = x.to(dtype=torch.float16)
            B = B.to(dtype=torch.float16)
            C = C.to(dtype=torch.float16)
            D = model.backbone['layers'][i]['mixer'].D.to(dtype=torch.float16)
            h[i] = h[i]._replace(ssm_state=h[i].ssm_state.to(dtype=torch.float16))

            # Save datas into .hex
            # if i == 0 and token_num == 0:
            #     print(dA.shape)
            #     print(dt.shape)
            #     print(x.shape)
            #     print(B.shape)
            #     print(C.shape)
            #     print(D.shape)
            #     print(h[i].ssm_state.shape)
            #     save_tensor_fp16_hex(dA,  os.path.join(save_dir, f"{i}_dA.hex"))
            #     save_tensor_fp16_hex(dt,  os.path.join(save_dir, f"{i}_dt.hex"))
            #     save_tensor_fp16_hex(x,   os.path.join(save_dir, f"{i}_x.hex"))
            #     save_tensor_fp16_hex(B,   os.path.join(save_dir, f"{i}_B.hex"))
            #     save_tensor_fp16_hex(C,   os.path.join(save_dir, f"{i}_C.hex"))
            #     save_tensor_fp16_hex(D,   os.path.join(save_dir, f"{i}_D.hex"))
            #     save_tensor_fp16_hex(h[i].ssm_state, os.path.join(save_dir, f"{i}_ssm_state.hex"))

            t1 = time.perf_counter()

            dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)

            h[i].ssm_state.copy_(h[i].ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)

            y = torch.einsum("bhpn, bn -> bhp", h[i].ssm_state, C)

            y = y + rearrange(D, "h -> h 1") * x
            # print("dtype of ssm_state:", h[i].ssm_state.dtype)
            # print("dtype of dA:", dA.dtype)
            # print("dtype of dBx:", dBx.dtype)
            y = rearrange(y, "b h p -> b (h p)")
            # if i == 0 and token_num == 0:
            #     save_tensor_fp16_hex(y,   os.path.join(save_dir, f"{i}_y.hex"))

            t2 = time.perf_counter()
            ssm_time.append(t2 - t1)
            # print("SSM time: ", ssm_time)
            y = y.to(dtype=torch.float32)
            
# ====================  SSM Block Finished  ====================

            y = model.backbone['layers'][i]['mixer'].norm(y, z)
            y = model.backbone['layers'][i]['mixer'].out_proj(y)

            residual = residual + y.unsqueeze(1)

            t2_m = time.perf_counter()
            mamba_time.append(t2_m - t1_m)

        average_list.append(sum(ssm_time) / len(ssm_time))
        average_list_m.append(sum(mamba_time) / len(mamba_time))
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
print(f'\nAVG SSM time:   {sum(average_list) / len(average_list)}\n')
print(f'\nAVG Mamba time: {sum(average_list_m) / len(average_list_m)}\n')