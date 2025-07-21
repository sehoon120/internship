import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from FXP_simulator import FXP16Simulator, FXP32Simulator, FXP8Simulator


class FastBiasedExp(nn.Module):
    def __init__(self):
        super().__init__()
        self.LOG2E = 1.4426950408889634  # 1 / ln(2)

    def forward(self, x):
        # 1. clip input to [-7, 0]
        x = torch.clamp(x, min=-7.0, max=0.0)

        # 2. convert exp(x) to 2^(x / ln(2)) = 2^(k + f)
        y = x * self.LOG2E  # convert to log2 domain

        k = torch.floor(y)                # integer part
        f = y - k                         # fractional part (in [0,1))

        # 3. 2^f ≈ 1 + f * ln(2) (1st-order approx)
        two_pow_f = 1 + f * 0.69314718    # ln(2)

        # 4. final result: 2^k * 2^f = (1 << k) * two_pow_f
        result = torch.ldexp(two_pow_f, k.to(torch.int32))  # 2^k * two_pow_f

        return result

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fxp8 = FXP8Simulator(frac_bits = 4)
    fxp16 = FXP16Simulator(frac_bits = 14)
    
    x = torch.linspace(-8, 1, 300)
    x_q = fxp8.quantize(x)
    x_dq = fxp8.dequantize(x_q)
    # print(x_dq)

    y_true = torch.exp(x)
    y_approx = FastBiasedExp()(x_dq)
    y_approx_q = fxp16.quantize(y_approx)
    y_approx_dq = fxp16.dequantize(y_approx_q)

    plt.plot(x.numpy(), y_true.numpy(), label="True exp(x)", linewidth=2)
    plt.plot(x.numpy(), y_approx_dq.detach().numpy(), label="FBEA Approx", linestyle="--")
    plt.axvline(x=-7, color='gray', linestyle=':', linewidth=0.8)
    plt.axvline(x=0, color='gray', linestyle=':', linewidth=0.8)
    plt.title("Fast Biased Exponential Approximation (clipped to [-7, 0])")
    plt.legend()
    plt.grid(True)
    plt.show()
