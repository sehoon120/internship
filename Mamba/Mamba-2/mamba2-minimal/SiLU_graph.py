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
        out[x < -7] = -0.0001

        mask0 = (x >= -7) & (x < -5)
        out[mask0] = -0.0135

        mask1 = (x >= -5) & (x < -1.5)
        out[mask1] = -0.06244 * x[mask1] - 0.3457

        mask2 = (x >= -1.5) & (x <= 0.75)
        out[mask2] = 0.232 * (x[mask2] + 1.181) ** 2 - 0.275

        mask3 = (x >= 0.75) & (x <= 4.448)
        out[mask3] = 1.05 * x[mask3] - 0.2781

        out[x > 4.448] = x[x > 4.448] - 0.0005

        return out

class ApproxSiLU16(nn.Module):
    def __init__(self):
        super().__init__()
        # 16 구간 (17개 경계점)
        self.segment = torch.linspace(-8.0, 6.0, steps=17)  # [-8.0, ..., 6.0]
        self.silu_vals = self.segment * torch.sigmoid(self.segment)  # 각 점의 정확한 SiLU 값

    def forward(self, x):
        out = torch.empty_like(x)

        # 6.0 이상인 경우: out = x - 0.0005
        mask_right = x > self.segment[-1]
        out[mask_right] = x[mask_right] - 0.0005

        # 나머지 값들은 선형 보간 수행
        x_clamped = torch.clamp(x, self.segment[0].item(), self.segment[-1].item())

        indices = torch.bucketize(x_clamped, self.segment) - 1
        indices = torch.clamp(indices, 0, len(self.segment) - 2)

        x0 = self.segment[indices]
        x1 = self.segment[indices + 1]
        y0 = self.silu_vals[indices]
        y1 = self.silu_vals[indices + 1]

        t = (x_clamped - x0) / (x1 - x0)
        out[~mask_right] = (y0 + t * (y1 - y0))[~mask_right]

        return out
    
import torch.nn.functional as F

if __name__ == '__main__':
    x = torch.linspace(-25, 25, 300)
    silu = nn.SiLU()
    approx = ApproxSiLU16()
    y_ref = silu(x)
    # y_ref = x * F.sigmoid(x)
    y_approx = approx(x)

    import matplotlib.pyplot as plt
    plt.plot(x.numpy(), y_ref.numpy(), label='SiLU (original)', linewidth=2)
    plt.plot(x.numpy(), y_approx.detach().numpy(), label='Approx SiLU', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.title("Comparison: SiLU vs Approximate")
    plt.show()
