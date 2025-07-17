import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. x 범위 정의
x_vals = torch.logspace(-4, 4, steps=500)
true_vals = 1.0 / torch.sqrt(x_vals)

# 2. 세그먼트 및 LUT 정의
segments = torch.tensor([0.0, 0.0015625, 0.0125, 0.5, 1, 2, 64, 1024, 24576])
lut_values = []
for i in range(len(segments)):
    val = 1.0 / torch.sqrt(segments[i] + 0.0001)
    lut_values.append(val)
lut = torch.tensor(lut_values)
print(lut)

# 3. LUT 기반 보간 함수
def interp_lut_inv_sqrt(x_vals, segments, lut):
    out = torch.zeros_like(x_vals)
    for i in range(len(segments) - 1):
        lower, upper = segments[i], segments[i + 1]
        mask = (x_vals >= lower) & (x_vals < upper)
        if mask.any():
            ratio = (x_vals[mask] - lower) / (upper - lower)
            out[mask] = lut[i] * (1 - ratio) + lut[i + 1] * ratio
    out[x_vals >= segments[-1]] = lut[-1]
    return out

approx_vals = interp_lut_inv_sqrt(x_vals, segments, lut)

# 4. 시각화
plt.figure(figsize=(10, 5))
plt.plot(x_vals, true_vals, label='True 1/sqrt(x)', linewidth=2)
plt.plot(x_vals, approx_vals, label='LUT Approximation', linestyle='--')
plt.xscale('log')
plt.xlabel('x (log scale)')
plt.ylabel('1/sqrt(x)')
plt.title('1/sqrt(x): True vs LUT Approximation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
