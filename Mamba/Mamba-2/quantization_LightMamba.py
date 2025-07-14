import torch
import scipy.linalg
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "mamba2-130m")

# === Step 1: Load Hugging Face Model Weights (FP32/FP16) ===
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("AntonV/mamba2-130m-hf")
model.save_pretrained(model_path + "/mamba2-130m-hf-raw", safe_serialization=False)


# === Step 2: Per-Channel Quantization (INT8) with PoT Scaling ===
def per_channel_quantize(tensor, num_bits=8):
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    scale = tensor.abs().max(dim=1, keepdim=True)[0] / qmax
    quantized = torch.round(tensor / scale).clamp(qmin, qmax)
    return quantized, scale

def pot_scale(scale):
    log2_scale = torch.log2(scale)
    pot = torch.round(log2_scale)
    return torch.pow(2, pot)


# === Step 3: Hadamard Transform (Rotation Fusion) ===
def hadamard_transform(x):
    size = x.shape[0]
    H = torch.from_numpy(scipy.linalg.hadamard(size)).float()
    return H @ x


# === Step 4: Tiling-aware Weight Reordering ===
def tile_reorder(weight, np, pp):
    num_heads, hidden_size = weight.shape
    reordered = []
    for h in range(0, num_heads, np):
        for hs in range(0, hidden_size, pp):
            tile = weight[h:h+np, hs:hs+pp]
            reordered.append(tile.flatten())
    return torch.cat(reordered)


# === Step 5: Save FPGA-Friendly INT8 Format ===
def save_fpga_ready_weight(weight, np, pp, file_path):
    # Apply Hadamard Transform (Rotation Fusion)
    rotated_weight = hadamard_transform(weight)

    # Quantize with PoT Scaling
    quantized_weight, scale = per_channel_quantize(rotated_weight)
    scale_pot = pot_scale(scale)

    # Tile Reordering
    reordered_weight = tile_reorder(quantized_weight, np, pp)

    # Save
    torch.save({
        "weight": reordered_weight,
        "scale_shift": torch.log2(scale_pot).int()
    }, file_path)


# === Example Usage ===
# Assume extracting one weight matrix from the model as example
weight = model.lm_head.weight.data.clone()  # Example: LM Head Weight
np, pp = 8, 64  # Example Tile Sizes (Adjust according to your FPGA design)
# np ≥ 2,pp ≥ 8
save_fpga_ready_weight(weight, np, pp, (model_path + "./mamba2-130m-fpga-ready.pt"))
