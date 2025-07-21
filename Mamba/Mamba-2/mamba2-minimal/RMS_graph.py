import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class InvSqrtApprox16Segment(nn.Module):
    def __init__(self):
        super().__init__()
        self.segment = torch.tensor([
            0.0001, 0.002, 0.004, 0.007,  # 0 근처 미세 분할
            0.01, 0.03, 0.1, 0.2, 0.3,                        # 중요 구간
            1.0, 2.0, 4.0, 8.0, 16.0,                            # 중간
            64.0, 1024.0                      # 평탄 구간
        ])
        self.lut = 1.0 / torch.sqrt(self.segment)

    def forward(self, x):
        out = torch.empty_like(x)
        x_clamped = torch.clamp(x, min=self.segment[0].item(), max=self.segment[-1].item())
        indices = torch.bucketize(x_clamped, self.segment) - 1
        indices = torch.clamp(indices, 0, len(self.segment) - 2)

        x0 = self.segment[indices]
        x1 = self.segment[indices + 1]
        y0 = self.lut[indices]
        y1 = self.lut[indices + 1]
        t = (x_clamped - x0) / (x1 - x0)

        out = y0 + t * (y1 - y0)
        return out



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # 테스트용 입력값 (로그 스케일까지 포함)
    x = torch.linspace(0.01, 70.0, steps=500)
    inv_sqrt_true = 1.0 / torch.sqrt(x)
    model = InvSqrtApprox16Segment()
    inv_sqrt_approx = model(x)
    abs_error = torch.abs(inv_sqrt_true - inv_sqrt_approx)

    # === 그래프 ===
    plt.figure(figsize=(12, 6))

    # 실제 함수 vs 근사
    plt.subplot(1, 2, 1)
    plt.plot(x.numpy(), inv_sqrt_true.numpy(), label='True 1/sqrt(x)', linewidth=2)
    plt.plot(x.numpy(), inv_sqrt_approx.detach().numpy(), '--', label='Approx (16-segment)', linewidth=2)
    plt.title('Function Comparison')
    plt.xlabel('x')
    plt.ylabel('1 / sqrt(x)')
    plt.legend()
    plt.grid(True)

    # 절대 오차
    plt.subplot(1, 2, 2)
    plt.plot(x.numpy(), abs_error.numpy(), color='red', linewidth=1)
    plt.title('Absolute Error')
    plt.xlabel('x')
    plt.ylabel('abs error')
    plt.grid(True)

    plt.tight_layout()
    plt.show()