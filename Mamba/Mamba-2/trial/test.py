from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import os

# ─── 설정 ───────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW    = 10        # 생성할 토큰 수
NUM_BLOCKS = 24
EMBED_DIM  = 768
DT_RANK    = 24
X_DIM      = 256
# ────────────────────────────────────────────────────────────────────

# ─── 유틸 함수 ──────────────────────────────────────────────────────
def RMSNorm(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

def conv1d(x, weight, bias, kernel_size=4):
    B, C = x.shape
    x = x.unsqueeze(-1)                      # (B,1,C)
    pad = kernel_size - 1
    x = F.pad(x, (pad, 0))                   # (B,1,C+pad)
    out = F.conv1d(x, weight, bias=bias,
                   stride=1, padding=0, groups=C)
    return out.squeeze(-1)                   # (B,C)
# ────────────────────────────────────────────────────────────────────

# ─── 가중치 로드 ─────────────────────────────────────────────────────
# state_dict 위치 (raw HF checkpoint)
base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sd = torch.load(os.path.join(base, "mamba2-130m/mamba2-130m-hf-raw/pytorch_model.bin"),
                map_location="cpu")

# 토크나이저만 HF에서 불러오고, 모델 블록과 헤드 가중치는 직접 사용
tokenizer  = AutoTokenizer.from_pretrained("AntonV/mamba2-130m-hf")
emb_w       = sd["backbone.embeddings.weight"].to(DEVICE)   # [50288,768]
normf_w     = sd["backbone.norm_f.weight"].to(DEVICE)       # [768]
lmhead_w    = sd["lm_head.weight"].to(DEVICE)               # [50288,768]
# ────────────────────────────────────────────────────────────────────

# ─── 한 레이어(블록) 계산 함수 ───────────────────────────────────────
def mamba_block(x_t, h_prev, layer_idx):
    prefix    = f"backbone.layers.{layer_idx}"
    # 1) weight 불러오기
    norm_w    = sd[f"{prefix}.norm.weight"].to(DEVICE)         # [768]
    inproj_w  = sd[f"{prefix}.mixer.in_proj.weight"].to(DEVICE)# [3352,768]
    mixnorm_w = sd[f"{prefix}.mixer.norm.weight"].to(DEVICE)   # [1536]
    A_log     = sd[f"{prefix}.mixer.A_log"].to(DEVICE)         # [24]
    D         = sd[f"{prefix}.mixer.D"].to(DEVICE)             # [24]
    dt_bias   = sd[f"{prefix}.mixer.dt_bias"].to(DEVICE)       # [24]
    outproj_w = sd[f"{prefix}.mixer.out_proj.weight"].to(DEVICE)# [768,1536]
    conv_w    = sd[f"{prefix}.mixer.conv1d.weight"].to(DEVICE) # [1792,1,4]
    conv_b    = sd[f"{prefix}.mixer.conv1d.bias"].to(DEVICE)   # [1792]

    # 2) Pre-Norm → SiLU
    x_norm = RMSNorm(x_t, norm_w)
    x_norm = F.silu(x_norm)

    # 3) in_proj → dt, B, C, X, Z
    proj = F.linear(x_norm, inproj_w)   # (B, 3352)
    dt, B, C, X, Z = torch.split(
        proj,
        [DT_RANK, EMBED_DIM, EMBED_DIM, X_DIM, 1536],
        dim=-1
    )
    Z = torch.sigmoid(Z)  # (B,1536)

    # 4) A_bar 계산
    Δ     = F.softplus(dt) + dt_bias    # (B,24)
    A     = -torch.exp(A_log)           # (24,)
    A_bar = torch.exp(Δ * A)            # (B,24)

    # 5) depthwise conv1d
    Bw, Cw, Xw = torch.split(conv_w, [EMBED_DIM, EMBED_DIM, X_DIM], dim=0)
    Bb, Cb, Xb = torch.split(conv_b, [EMBED_DIM, EMBED_DIM, X_DIM], dim=0)
    B_conv = conv1d(B, Bw, Bb)  # (B,768)
    C_conv = conv1d(C, Cw, Cb)  # (B,768)
    X_conv = conv1d(X, Xw, Xb)  # (B,256)

    # 6) 상태업데이트: h = A_bar·h_prev + B_conv
    tile = EMBED_DIM // DT_RANK  # 32
    A_exp = A_bar.repeat_interleave(tile, dim=-1)  # (B,768)
    h_new = A_exp * h_prev + B_conv               # (B,768)

    # 7) y = h·C_conv + D·X_conv
    D_exp = D.repeat_interleave(tile).unsqueeze(0)                  # (1,768)
    X_exp = X_conv.repeat_interleave(EMBED_DIM//X_DIM, dim=1)      # (B,768)
    y     = h_new * C_conv + D_exp * X_exp                         # (B,768)

    # 8) gated out & out_proj
    cat  = torch.cat([y, X_exp], dim=-1)           # (B,1536)
    gn   = RMSNorm(cat, mixnorm_w) * Z            # (B,1536)
    out  = F.linear(gn, outproj_w)                # (B,768)

    # 최종 residual
    return out + x_t, h_new
# ────────────────────────────────────────────────────────────────────

# ─── 1) 프롬프트 처리 (state 채우기) ────────────────────────────────
prompt     = "Hey how are you doing?"
ids        = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
batch, L   = ids.shape

# hidden state 리스트 초기화
h_list = [torch.zeros(batch, EMBED_DIM, device=DEVICE) for _ in range(NUM_BLOCKS)]

# prompt embedding + 블록 순차 업데이트
with torch.no_grad():
    x_seq = F.embedding(ids, emb_w)  # (B, L, 768)
    for t in range(L):
        x_t = x_seq[:, t, :]
        for l in range(NUM_BLOCKS):
            x_t, h_list[l] = mamba_block(x_t, h_list[l], l)
# ────────────────────────────────────────────────────────────────────

# ─── 2) 수동 생성 루프 ───────────────────────────────────────────────
generated = ids.clone()  # (B, cur_len)
with torch.no_grad():
    for _ in range(MAX_NEW):
        # (1) 마지막 토큰 임베딩
        last_id = generated[:, -1:]                     # (B,1)
        x_t     = F.embedding(last_id, emb_w).squeeze(1) # (B,768)

        # (2) 블록별 state 업데이트
        for l in range(NUM_BLOCKS):
            x_t, h_list[l] = mamba_block(x_t, h_list[l], l)

        # # (3) lm_head → 다음 토큰
        # logits = F.linear(RMSNorm(x_t, normf_w), lmhead_w)  # (B, vocab_size)
        # probs  = F.softmax(logits / 1.0, dim=-1)             # temperature=1.0
        # next_id = torch.multinomial(probs, num_samples=1)   # (B,1)

        # # (4) append
        # generated = torch.cat([generated, next_id], dim=1)

        # (3) logits → probs → 샘플링
        logits = F.linear(RMSNorm(x_t, normf_w), lmhead_w)    # (B, V)
        logits = logits.clamp(-20.0, 20.0)

        # 1) stable softmax: 최대값 빼 주기
        max_logits = logits.max(dim=-1, keepdim=True).values
        stable_logits = logits - max_logits                   # (B, V)

        # 2) exp → 정규화
        probs = stable_logits.exp()
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # 3) 혹시 남아 있는 nan/inf/음수를 0으로 바꿔 주기
        probs = probs.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=0.0)

        # 4) sampling
        next_id = torch.multinomial(probs, num_samples=1)     # (B,1)

# ────────────────────────────────────────────────────────────────────

# ─── 3) 결과 출력 ──────────────────────────────────────────────────
print(tokenizer.decode(generated[0], skip_special_tokens=True))
# ────────────────────────────────────────────────────────────────────
