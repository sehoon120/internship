import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ApproxSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.empty_like(x)

        # 각 구간별 연산 정의
        out[x < -5] = -0.0135

        mask1 = (x >= -5) & (x < -1.5)
        out[mask1] = -0.06244 * x[mask1] - 0.3457

        mask2 = (x >= -1.5) & (x <= 0.75)
        out[mask2] = 0.232 * (x[mask2] + 1.181) ** 2 - 0.275

        out[x > 0.75] = 1.05 * x[x > 0.75] - 0.2781

        return out




x = torch.linspace(-6, 3, 100)
silu = nn.SiLU()
approx = ApproxSiLU()

y_ref = silu(x)
y_approx = approx(x)

import matplotlib.pyplot as plt
plt.plot(x.numpy(), y_ref.numpy(), label='SiLU (original)', linewidth=2)
plt.plot(x.numpy(), y_approx.detach().numpy(), label='Approx SiLU', linestyle='--')
plt.legend()
plt.grid(True)
plt.title("Comparison: SiLU vs Approximate")
plt.show()
