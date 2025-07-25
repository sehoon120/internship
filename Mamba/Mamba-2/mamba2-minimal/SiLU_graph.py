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
    
class ApproxSiLU16_FXP(nn.Module):
    def __init__(self, in_frac=11, out_frac=10):
        super().__init__()
        self.in_frac = in_frac
        self.out_frac = out_frac

        # 16 segments: 17 points
        self.segment_fp = torch.linspace(-8.0, 6.0, steps=17)
        self.silu_fp = self.segment_fp * torch.sigmoid(self.segment_fp)

        # FXP 정수 테이블로 변환
        self.segment = (self.segment_fp * (1 << in_frac)).round().to(torch.int16)     # FXP16.11
        self.silu_vals = (self.silu_fp * (1 << out_frac)).round().to(torch.int16)     # FXP16.10

    def forward(self, x):
        # x: float32지만 값은 FXP16.11로 표현된 값
        x_int = (x * (1 << self.in_frac)).round().to(torch.int32)

        out_int = torch.empty_like(x_int)

        # 경계 처리: x > max segment → out = x - 0.0005 ≈ x (FXP 그대로 사용)
        max_seg = self.segment[-1].item()
        mask_right = x_int > max_seg
        out_int[mask_right] = (x_int[mask_right] >> (self.in_frac - self.out_frac))  # FXP16.11 → FXP16.10

        # Clamp x to valid segment range
        x_clamped = torch.clamp(x_int, self.segment[0].item(), self.segment[-1].item())

        # Bucket index 구하기
        seg = self.segment.to(x_int.device)
        idx = torch.bucketize(x_clamped, seg, right=False) - 1  # [0, 15]
        idx = torch.clamp(idx, 0, len(seg) - 2)

        # 구간 점들
        x0 = seg[idx]             # FXP16.11 int
        x1 = seg[idx + 1]         # FXP16.11 int
        y0 = self.silu_vals[idx]     # FXP16.10 int
        y1 = self.silu_vals[idx + 1] # FXP16.10 int

        # 보간 비율 t = (x - x0) / (x1 - x0), fixed-point로 계산
        dx = x_clamped - x0
        dx_total = x1 - x0
        eps = 1  # avoid division by zero
        dx_total = torch.clamp(dx_total, min=eps)

        t_fx = ((dx << self.out_frac) + (dx_total // 2)) // dx_total  # t_fx: FXP16.10

        # y = y0 + t * (y1 - y0)
        dy = y1 - y0
        interp = y0 + ((t_fx * dy + (1 << (self.out_frac - 1))) >> self.out_frac)  # FXP16.10

        out_int[~mask_right] = interp[~mask_right]

        # 정수형 → float로 변환하여 반환 (여전히 FXP16.10 형식)
        return out_int.to(torch.float32) / (1 << self.out_frac)

    
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
