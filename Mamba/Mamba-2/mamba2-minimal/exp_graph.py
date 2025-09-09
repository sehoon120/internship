import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from FXP_simulator import FXP16Simulator, FXP32Simulator, FXP8Simulator



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exp_fast8(x: torch.Tensor) -> torch.Tensor:
    """
    8-segment PWL 근사로 e^x 계산.
    e^x = 2^(x*log2(e)) = 2^(k+f) = 2^k * 2^f
    2^f는 f∈[0,1)에서 8구간(pwl) 1차 근사.
    """
    # 상수/준비
    log2e = 1.4426950408889634  # 1/ln(2)
    device, dtype = x.device, x.dtype

    # t = x*log2(e) = k + f
    t = x * torch.tensor(log2e, dtype=dtype, device=device)
    k = torch.floor(t)                          # 정수부 (float)
    f = t - k                                   # 소수부 ∈ [0,1)

    # 8 구간 인덱스와 좌측 경계 f0
    seg = torch.clamp((f * 8).to(torch.int64), 0, 7)
    f0  = seg.to(dtype) / 8.0

    # 구간 끝점 값: y0=2^(f0), y1=2^(f0+1/8)
    # (테이블 사전계산)
    with torch.no_grad():
        boundaries = torch.arange(9, device=device, dtype=dtype) / 8.0  # 0,1/8,...,1
        pow2_table = torch.pow(torch.tensor(2.0, dtype=dtype, device=device), boundaries)
    y0 = pow2_table[seg]              # 2^(seg/8)
    y1 = pow2_table[seg + 1]          # 2^((seg+1)/8)

    # 선형 근사: 2^f ≈ y0 + slope*(f - f0)
    # slope = (y1 - y0) / (1/8) = 8*(y1 - y0)
    slope = 8.0 * (y1 - y0)
    two_pow_f = y0 + slope * (f - f0)

    # 최종: 2^k * 2^f  (ldexp: mantissa * 2^exponent)
    # k는 float이므로 정수로 변환
    k_int = k.to(torch.int32)
    y = torch.ldexp(two_pow_f, k_int)
    return y

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
    
class ApproxExp16_FXP(nn.Module):
    def __init__(self, in_frac=13, out_frac=16):
        super().__init__()
        self.in_frac = in_frac
        self.out_frac = out_frac

        # Input range: clip between [-10, 4] (exp(-10) ≈ 0, exp(4) ≈ 54.6)
        self.x_pts_fp = torch.linspace(-10, 4, steps=17)  # 16 segments
        self.exp_vals_fp = torch.exp(self.x_pts_fp)

        # FXP 버전으로 저장
        self.x_pts = (self.x_pts_fp * (1 << in_frac)).round().to(torch.int32)
        self.exp_vals = (self.exp_vals_fp * (1 << out_frac)).round().to(torch.int64)

    def forward(self, x):
        x_int = (x * (1 << self.in_frac)).round().to(torch.int32)

        out_int = torch.empty_like(x_int, dtype=torch.int64)

        # 너무 작으면 0에 수렴 (exp(-inf) ≈ 0)
        min_x = self.x_pts[0].item()
        max_x = self.x_pts[-1].item()

        mask_low = x_int <= min_x
        mask_high = x_int >= max_x

        out_int[mask_low] = 0
        out_int[mask_high] = self.exp_vals[-1]

        # 나머지에 대해 보간 수행
        x_clamped = torch.clamp(x_int, min_x, max_x)

        # segment index
        idx = torch.bucketize(x_clamped, self.x_pts.to(x.device), right=False) - 1
        idx = torch.clamp(idx, 0, len(self.x_pts) - 2)

        x0 = self.x_pts.to(x.device)[idx]
        x1 = self.x_pts.to(x.device)[idx + 1]
        y0 = self.exp_vals.to(x.device)[idx]
        y1 = self.exp_vals.to(x.device)[idx + 1]

        dx = x_clamped - x0
        dx_total = x1 - x0
        dx_total = torch.clamp(dx_total, min=1)

        # 보간 계수: t_fx = (dx / dx_total) in FXP.out_frac
        t_fx = ((dx << self.out_frac) + (dx_total // 2)) // dx_total  # FXP.out_frac

        # 보간: y = y0 + t * (y1 - y0)
        dy = y1 - y0
        interp = y0 + ((t_fx * dy + (1 << (self.out_frac - 1))) >> self.out_frac)

        out_int[~(mask_low | mask_high)] = interp[~(mask_low | mask_high)]

        # 부호 반영: A = -exp(...)
        return -out_int.to(torch.float32) / (1 << self.out_frac)

class ApproxExp_FXP32in16out14(nn.Module):
    def __init__(self, in_frac=16, out_frac=14):
        super().__init__()
        self.in_frac = in_frac
        self.out_frac = out_frac

        # 실수 기준 점들: 16개 구간 → 17개 포인트
        self.x_pts_fp = torch.linspace(-10.0, 4.0, steps=17)  # [-10, ..., 4]
        self.exp_vals_fp = torch.exp(self.x_pts_fp)

        # FXP 정수 테이블로 변환
        self.x_pts = (self.x_pts_fp * (1 << in_frac)).round().to(torch.int32)      # FXP32.16
        self.exp_vals = (self.exp_vals_fp * (1 << out_frac)).round().to(torch.int32)  # FXP16.14

    def forward(self, x):
        """
        x: float32 값이지만 의미적으로는 FXP32.16 값 (e.g., -32768.0 ~ +32768.0)
        반환: FXP16.14 float 값 (실제 정수 기반)
        """
        x_pts = self.x_pts.to(x.device)
        exp_vals = self.exp_vals.to(x.device)
        x_int = (x * (1 << self.in_frac)).round().to(torch.int32)
        out_int = torch.empty_like(x_int)

        # 경계 처리
        min_x, max_x = x_pts[0].item(), x_pts[-1].item()
        mask_low = x_int <= min_x
        mask_high = x_int >= max_x
        x_clamped = torch.clamp(x_int, min_x, max_x)

        # 고정 구간 인덱싱
        idx = torch.bucketize(x_clamped, x_pts) - 1
        idx = torch.clamp(idx, 0, len(x_pts) - 2)

        x0 = x_pts[idx]
        x1 = x_pts[idx + 1]
        y0 = exp_vals[idx]
        y1 = exp_vals[idx + 1]

        dx = x_clamped - x0
        dx_total = x1 - x0
        dx_total = torch.clamp(dx_total, min=1)

        # 보간 계수 t_fx (FXP.out_frac)
        t_fx = ((dx << self.out_frac) + (dx_total // 2)) // dx_total  # rounding 보간

        # 보간 적용: y = y0 + t * (y1 - y0)
        dy = y1 - y0
        interp = y0 + ((t_fx * dy + (1 << (self.out_frac - 1))) >> self.out_frac)

        # 경계 보정
        out_int[mask_low] = exp_vals[0]
        out_int[mask_high] = exp_vals[-1]
        out_int[~(mask_low | mask_high)] = interp[~(mask_low | mask_high)]

        # 정수 기반 FXP16.14 float 형태로 반환
        return out_int.to(torch.float32) / (1 << self.out_frac)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # fxp8 = FXP8Simulator(frac_bits = 4)
    # fxp16 = FXP16Simulator(frac_bits = 14)
    # fxp16_13 = FXP16Simulator(frac_bits = 13)
    # fxp32 = FXP32Simulator(frac_bits = 16)
    
    x = torch.linspace(-10, 7, 500)
    # x_q = fxp8.quantize(x)
    # x_dq = fxp8.dequantize(x_q)
    # print(x_dq)

    '''
    # ==================== exp ====================
    y_true = torch.exp(x)
    # --- 기본 근사 ---
    y_approx = exp_fast8(x)
    # ==================================================
    '''

    # import numpy as np
    # import struct

    # def f2h(x):  # float32 -> float16 hex (IEEE-754)
    #     return struct.unpack('<H', np.float16(x).tobytes())[0]

    # y0 = [2**(i/8.0) for i in range(8)]
    # y1 = [2**((i+1)/8.0) for i in range(8)]
    # slope = [8.0*(y1[i]-y0[i]) for i in range(8)]

    # print("// --- paste into pwl8_rom ---")
    # for i in range(8):
    #     print(f"3'd{i}: begin y0_o <= 16'h{f2h(y0[i]):04X}; slope_o <= 16'h{f2h(slope[i]):04X}; end")


    # '''
    # ==================== Softplus ====================
    y_true = F.softplus(x)
    # --- 대칭 근사 ---
    # y_approx = torch.where(x <= 0, torch.exp(x), x + torch.exp(-x))
    # --- 기본 근사 --- 
    ln2 = math.log(2)
    y_approx = torch.where(x == 0, ln2, x / (1 - exp_fast8(-x/ln2)))
    # ==================================================
    # '''

    # y_approx = FastBiasedExp()(x_dq)
    # x_q_16_13 = fxp16_13.quantize(x)
    # y_approx1 = ApproxExp16_FXP(in_frac=13, out_frac=16)
    # y_32_16 = y_approx1(x_q_16_13)
    # y1 = fxp32.dequantize(y_32_16)

    # x_q_32_16 = fxp32.quantize(x)
    # print(x_q_32_16)
    # y_approx2 = ApproxExp_FXP32in16out14(in_frac=16, out_frac=14)  
    # y_16_14 = y_approx2(x_q_32_16)
    # y2 = fxp16.dequantize(y_16_14)




    plt.plot(x.numpy(), y_true.numpy(), label="True exp(x)", linewidth=2)
    plt.plot(x.numpy(), y_approx.numpy(), label="FBEA Approx", linestyle="--")
    plt.axvline(x=-7, color='gray', linestyle=':', linewidth=0.8)
    plt.axvline(x=0, color='gray', linestyle=':', linewidth=0.8)
    plt.title("Fast Biased Exponential Approximation (clipped to [-7, 0])")
    plt.legend()
    plt.grid(True)
    plt.show()
