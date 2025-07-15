import torch
import scipy.linalg
import os
import math


# Hadamard like matrix로 바꿔서 다시만들어보기


current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
model_path = os.path.join(base_dir, "mamba2-130m")
# print(model_path)

# === Step 1: Load Hugging Face Model Weights (FP32/FP16) ===
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("AntonV/mamba2-130m-hf")
model.save_pretrained(model_path + "/mamba2-130m-hf-raw", safe_serialization=False)

print('\n==================================================\n')

# for name, param in model.named_parameters():
#     print(name, param.shape)

# backbone.embeddings.weight torch.Size([50288, 768])
# backbone.layers.0.norm.weight torch.Size([768])
# backbone.layers.0.mixer.dt_bias torch.Size([24])
# backbone.layers.0.mixer.A_log torch.Size([24])
# backbone.layers.0.mixer.D torch.Size([24])
# backbone.layers.0.mixer.conv1d.weight torch.Size([1792, 1, 4])
# backbone.layers.0.mixer.conv1d.bias torch.Size([1792])
# backbone.layers.0.mixer.in_proj.weight torch.Size([3352, 768])
# backbone.layers.0.mixer.norm.weight torch.Size([1536])
# backbone.layers.0.mixer.out_proj.weight torch.Size([768, 1536])
# backbone.norm_f.weight torch.Size([768])



print('\n==================================================\n')

def hadamard_transform_QW(x, tile_size=64):
    seq_len, dim = x.shape
    if seq_len % tile_size != 0:
        pad_len = tile_size - (seq_len % tile_size)
        pad = torch.zeros((pad_len, dim), dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=0)
    padded_len = x.shape[0]

    # Apply Hadamard on rows (dimension 0)
    x_reshaped = x.view(-1, tile_size, dim)  # [batch, tile_size, dim]
    H = torch.from_numpy(scipy.linalg.hadamard(tile_size)).float().to(x.device)  # [tile_size, tile_size]
    
    # Apply Hadamard to each [tile_size, dim] -> [tile_size, dim]
    rotated = torch.matmul(H, x_reshaped)  # [batch, tile_size, dim]
    
    return rotated.view(padded_len, dim)[:seq_len]

def hadamard_rotate_WQ(x, tile_size=64):
    vocab_size, dim = x.shape
    remainder = dim % tile_size
    if remainder != 0:
        pad_len = tile_size - remainder
        pad = torch.zeros((vocab_size, pad_len), dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=1)
        dim += pad_len

    x_reshaped = x.view(vocab_size, -1, tile_size)  # [V, dim//tile, tile]
    H = torch.from_numpy(scipy.linalg.hadamard(tile_size)).float().to(x.device)  # [tile, tile]

    rotated = torch.matmul(x_reshaped, H)  # [V, dim//tile, tile]
    return rotated.view(vocab_size, dim)[:, :dim - pad_len if remainder != 0 else dim]

def hadamard_in_projection(W_in, gamma):
    out_dim, hidden_dim = W_in.shape
    next_pow2 = 2 ** (hidden_dim - 1).bit_length()

    if hidden_dim != next_pow2:
        pad_len = next_pow2 - hidden_dim
        W_in = torch.nn.functional.pad(W_in, (0, pad_len))       # pad right columns
        gamma = torch.nn.functional.pad(gamma, (0, pad_len))
        hidden_dim = next_pow2

    # scale
    W_scaled = W_in * gamma  # [3352, hidden_dim]

    # rotate
    H = torch.from_numpy(scipy.linalg.hadamard(hidden_dim)).float().to(W_in.device)
    W_rotated = torch.matmul(W_scaled, H.T)

    return W_rotated[:, :768]  # return trimmed output

def double_rotate_out_proj(W_out, H_dim=768, Q_dim=1536):
    """
    Perform: W = H^T · W_out · Q
    - W_out: [768, 1536]
    - H: Hadamard matrix for rows (768 x 768)
    - Q: Hadamard matrix for columns (1536 x 1536)
    """
    # Check power-of-2
    if H_dim & (H_dim - 1) != 0 or Q_dim & (Q_dim - 1) != 0:
        raise ValueError("Both dimensions must be powers of 2 for Hadamard")

    # Hadamard matrices
    H = torch.from_numpy(scipy.linalg.hadamard(H_dim)).float().to(W_out.device)
    Q = torch.from_numpy(scipy.linalg.hadamard(Q_dim)).float().to(W_out.device)

    # Apply: H^T @ W @ Q
    W_rotated = torch.matmul(H.T, torch.matmul(W_out, Q))

    return W_rotated

def hadamard_like_rotate(x):
    """
    Apply Hadamard-like rotation for non-power-of-2 dimension.
    x: Tensor of shape [..., dim] (last dim = feature dim)
    """
    *batch_dims, dim = x.shape

    # Generate a fixed pseudo-Hadamard-like matrix with ±1
    torch.manual_seed(42)  # Fixed seed for reproducibility
    H_like = torch.randint(0, 2, (dim, dim)).float().to(x.device)
    H_like[H_like == 0] = -1  # make elements ±1

    # Normalize rows to roughly preserve scale
    H_like = H_like / (dim ** 0.5)

    # Apply matrix multiplication
    x_flat = x.view(-1, dim)         # [..., dim] → [N, dim]
    x_rot = torch.matmul(x_flat, H_like.T)
    return x_rot.view(*batch_dims, dim)

# ====================================================================================================

def rms_normalize(x, eps=1e-6):
    # x.shape: [batch_size, seq_len, dim]
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    x_norm = x / rms
    return x_norm

def scale_with_gamma(x_norm, gamma):
    # gamma.shape: [dim]
    return x_norm * gamma  # broadcasting along batch and seq dims


print('\n==================================================\n')

# weight_embedding = dict(model.named_parameters())['backbone.embeddings.weight'] # torch.Size([50288, 768])
# rotated_weight_embedding = hadamard_rotate_WQ(weight_embedding)

# x_norm = rms_normalize(x)  # step (1)
# gamma = model.backbone.layers[0].norm.weight.data  # step (2)
# # y = scale_with_gamma(x_norm, gamma) # original
# W_in = model.backbone.layers[0].mixer.in_proj.weight.data
# weight_inproj = hadamard_in_projection(W_in, gamma)
# print(weight_inproj)
# print(weight_inproj.shape)

W_out = model.backbone.layers[0].mixer.out_proj.weight.data  # [768, 1536]
weight_outproj = double_rotate_out_proj(W_out)
print(weight_outproj)
print(weight_outproj.shape)

# weight_lmhead = model.lm_head.weight.data.clone() # torch.Size([50288, 768])
# rotated_weight_lmhead = hadamard_transform_QW(weight_lmhead)

print('\n==================================================\n')


