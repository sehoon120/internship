from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import math
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
model_path = os.path.join(base_dir, "mamba2-130m")

tokenizer = AutoTokenizer.from_pretrained("AntonV/mamba2-130m-hf")
model = AutoModelForCausalLM.from_pretrained("AntonV/mamba2-130m-hf")

# print('\n==================================================\n')
# print(model)
# print('\n==================================================\n')

input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"]
# out = model.generate(input_ids, max_new_tokens=10)
# print(tokenizer.batch_decode(out))
batch_size, seq_len = input_ids.shape

print('\n==================================================\n')

# Load Hugging Face weights
state_dict = torch.load(os.path.join(model_path, "mamba2-130m-hf-raw", "pytorch_model.bin"), map_location="cpu")

# === 설정 ===
NUM_BLOCKS = 24
EMBED_DIM = 768
DT_RANK = 24
X_DIM = 256
OUT_PROJ_IN = 1536

# === RMSNorm 구현 ===
def RMSNorm(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

# === conv1d 구현 ===
def conv1d(x, weight, bias, kernel_size=4):
    B, C = x.shape
    x = x.unsqueeze(-1)  # (B, 1, C)

    # Causal padding on the left: (kernel_size - 1) zeros
    pad_len = kernel_size - 1
    x_padded = F.pad(x, (pad_len, 0))  # pad left only → (B, 1, C+pad)

    # Make weight usable: (C, 1, K) → grouped by channels
    out = F.conv1d(x_padded, weight, bias=bias, stride=1, padding=0, groups=C)  # (B, C, 1)
    return out.squeeze(-1)  # (B, C)

h_list = [torch.zeros(batch_size, EMBED_DIM) for _ in range(NUM_BLOCKS)]
all_logits = []

embedding_weight = state_dict["backbone.embeddings.weight"]  # [50288, 768]
norm_f_weight = state_dict["backbone.norm_f.weight"]  # [768]
lm_head_weight = state_dict["lm_head.weight"]  # [50288, 768]

x = F.embedding(input_ids, embedding_weight)  # (1, 3, 768)
x_t = None

for t in range(seq_len):
    x_t = x[:, t, :]  # 현재 토큰

    for layer in range(NUM_BLOCKS):
        x_residual = x_t

        prefix = f"backbone.layers.{layer}"
        norm_w = state_dict[f"{prefix}.norm.weight"]  # [768]
        in_proj_w = state_dict[f"{prefix}.mixer.in_proj.weight"]  # [3352, 768]
        mixer_norm_w = state_dict[f"{prefix}.mixer.norm.weight"]  # [1536]
        A_log = state_dict[f"{prefix}.mixer.A_log"]  # [24]
        D = state_dict[f"{prefix}.mixer.D"]  # [24]
        dt_bias = state_dict[f"{prefix}.mixer.dt_bias"]  # [24]
        out_proj_w = state_dict[f"{prefix}.mixer.out_proj.weight"]  # [768, 1536]
        conv1d_w = state_dict[f"{prefix}.mixer.conv1d.weight"]  # [1792, 1, 4]
        conv1d_b = state_dict[f"{prefix}.mixer.conv1d.bias"]  # [1792]
    
        # 1. Pre-Norm
        x_norm = RMSNorm(x_t, norm_w)

        # SiLU
        x_norm = F.silu(x_norm)

        # 2. in_proj(x) → [dt, B, C, X] with shape split from 3352
        in_proj = F.linear(x_norm, in_proj_w)  # (1, 3352)
        dt, B, C, X, Z = torch.split(in_proj, [DT_RANK, EMBED_DIM, EMBED_DIM, 256, 1536], dim=-1)
        Z = F.sigmoid(Z)  # [B, 1536]

        # 3. dt → softplus(Δ), A_log + dt_bias → A
        Δ = F.softplus(dt) + dt_bias  # (1, 24)
        A = -torch.exp(A_log)       # (24,)
        A_bar = torch.exp(Δ * A)    # (1, 24)
        
        # 4. B, C, X → conv1d (depthwise, padding=3)
        B_w, C_w, X_w = torch.split(conv1d_w, [EMBED_DIM, EMBED_DIM, X_DIM], dim=0)
        B_b, C_b, X_b = torch.split(conv1d_b, [EMBED_DIM, EMBED_DIM, X_DIM], dim=0)


        # B_flat = B.view(-1, EMBED_DIM)                # (B·L, D)
        # B_conv = conv1d(B_flat, B_w, B_b).view(batch_size, seq_len, -1)

        B_conv = conv1d(B, B_w, B_b)  # (1,768)
        C_conv = conv1d(C, C_w, C_b)
        X_conv = conv1d(X, X_w, X_b)

        # 5. h_t = A * h_prev + B_conv with broadcast
        tile = EMBED_DIM // DT_RANK  # 768//24 = 32
        A_bar_expanded = A_bar.repeat_interleave(tile, dim=-1)
        h_list[layer] = A_bar_expanded * h_list[layer] + B_conv

        # 6. y_t = h_t * C_conv + D * X_conv
        # print(h_list[layer].shape)
        # print(C_conv.shape)
        # print(D.shape)
        # print(X_conv.shape)
        D_expanded = D.repeat_interleave(EMBED_DIM // DT_RANK).unsqueeze(0)  # D: (24,) → (1,768)
        X_expanded = X_conv.repeat_interleave(EMBED_DIM // X_DIM, dim=1)  # X_conv: (1,256) → (1,768)
        # y = h_list[layer] * C_conv + D * X_conv
        y = h_list[layer] * C_conv + D_expanded * X_expanded

        # 7. out_input = concat(y_t, X_conv) → (1536)
        cat = torch.cat([y, X_expanded], dim=-1)  # (1, 1536)
        y_cat_norm = RMSNorm(cat, mixer_norm_w)  # Gated RMSNorm
        gated = y_cat_norm * Z

        # 8. norm(out_input) → out_proj
        x_out = F.linear(gated, out_proj_w)  # (1, 768)
        x_t = x_out + x_residual

    logit_t = F.linear(RMSNorm(x_t, norm_f_weight), lm_head_weight)  # (1, 50288)
    all_logits.append(logit_t)

logits_seq = torch.cat(all_logits, dim=0).unsqueeze(0)  # (1, seq_len, vocab_size)

generated_ids = torch.argmax(logits_seq, dim=-1)  # (1, seq_len)
decoded = tokenizer.batch_decode(generated_ids[0])
print("Decoded:", decoded)
