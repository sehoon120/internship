import torch
ckpt = torch.load(r"C:\Internship\mamba2-2.7b\mamba2_2.7b_quantized.pth", map_location="cpu")
print(ckpt.keys())
