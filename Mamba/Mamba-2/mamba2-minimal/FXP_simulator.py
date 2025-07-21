import torch
import torch.nn.functional as F

class FXP8Simulator:
    def __init__(self, frac_bits=6):
        self.total_bits = 8
        self.frac_bits = frac_bits
        self.scale = 2 ** frac_bits
        self.qmin = -2 ** (self.total_bits - 1)
        self.qmax = 2 ** (self.total_bits - 1) - 1

    def quantize(self, x_float):
        return (torch.round(x_float * self.scale)
                .clamp(self.qmin, self.qmax)
                .to(torch.int8))

    def dequantize(self, x_int):
        return x_int.to(torch.float32) / self.scale

    def add(self, x, y):
        return (x.to(torch.int16) + y.to(torch.int16)).clamp(self.qmin, self.qmax).to(torch.int8)

    def mul(self, x, y):
        prod = x.to(torch.int16) * y.to(torch.int16)
        return (prod >> self.frac_bits).clamp(self.qmin, self.qmax).to(torch.int8)

    def fxp_matmul(self, A, B):
        prod = torch.matmul(A.to(torch.int16), B.to(torch.int16))
        return (prod >> self.frac_bits).clamp(self.qmin, self.qmax).to(torch.int8)

    def fxp_einsum(self, equation, *operands):
        if equation == 'i,i->':
            x, y = operands
            prod = x.to(torch.int16) * y.to(torch.int16)
            summed = prod.sum()
            return (summed >> self.frac_bits).clamp(self.qmin, self.qmax).to(torch.int8)
        else:
            raise NotImplementedError(f"Equation '{equation}' not supported.")

    def fxp_conv1d(self, x: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0):
        x_i16 = x.to(torch.int16)
        w_i16 = weight.to(torch.int16)
        out = F.conv1d(x_i16, w_i16, bias=None, stride=stride, padding=padding)
        out = (out >> self.frac_bits).clamp(self.qmin, self.qmax).to(torch.int8)
        if bias is not None:
            out += bias.to(torch.int8).view(1, -1, 1)
        return out

    def fxp_conv2d(self, x: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0):
        x_i16 = x.to(torch.int16)
        w_i16 = weight.to(torch.int16)
        out = F.conv2d(x_i16, w_i16, bias=None, stride=stride, padding=padding)
        out = (out >> self.frac_bits).clamp(self.qmin, self.qmax).to(torch.int8)
        if bias is not None:
            out += bias.to(torch.int8).view(1, -1, 1, 1)
        return out

    def fxp_sum(self, x: torch.Tensor):
        x_i16 = x.to(torch.int16)
        total = x_i16.sum()
        return total.clamp(self.qmin, self.qmax).to(torch.int8)

class FXP16Simulator:
    def __init__(self, frac_bits=14):
        self.total_bits = 16
        self.frac_bits = frac_bits
        self.scale = 2 ** frac_bits
        self.qmin = -2 ** (self.total_bits - 1)
        self.qmax = 2 ** (self.total_bits - 1) - 1

    def quantize(self, x_float):
        return (torch.round(x_float * self.scale)
                .clamp(self.qmin, self.qmax)
                .to(torch.int16))

    def dequantize(self, x_int):
        return x_int.to(torch.float32) / self.scale

    def add(self, x, y):
        return (x.to(torch.int32) + y.to(torch.int32)).clamp(self.qmin, self.qmax).to(torch.int16)

    def mul(self, x, y):
        prod = x.to(torch.int32) * y.to(torch.int32)
        return (prod >> self.frac_bits).clamp(self.qmin, self.qmax).to(torch.int16)

    def fxp_matmul(self, A, B):
        prod = torch.matmul(A.to(torch.int32), B.to(torch.int32))
        return (prod >> self.frac_bits).clamp(self.qmin, self.qmax).to(torch.int16)

    def fxp_einsum(self, equation, *operands):
        if equation == 'i,i->':
            x, y = operands
            prod = x.to(torch.int32) * y.to(torch.int32)
            summed = prod.sum()
            return (summed >> self.frac_bits).clamp(self.qmin, self.qmax).to(torch.int16)
        else:
            raise NotImplementedError(f"Equation '{equation}' not supported.")
        
    def fxp_conv1d(self, x: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0):
        """ FXP Conv1D 시뮬레이션 """
        x_i32 = x.to(torch.int32)
        w_i32 = weight.to(torch.int32)
        out = F.conv1d(x_i32, w_i32, bias=None, stride=stride, padding=padding)
        out = (out >> self.frac_bits).clamp(self.qmin, self.qmax).to(torch.int16)
        if bias is not None:
            out += bias.to(torch.int16).view(1, -1, 1)
        return out

    def fxp_conv2d(self, x: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0):
        """ FXP Conv2D 시뮬레이션 """
        x_i32 = x.to(torch.int32)
        w_i32 = weight.to(torch.int32)
        out = F.conv2d(x_i32, w_i32, bias=None, stride=stride, padding=padding)
        out = (out >> self.frac_bits).clamp(self.qmin, self.qmax).to(torch.int16)
        if bias is not None:
            out += bias.to(torch.int16).view(1, -1, 1, 1)
        return out
    
    def fxp_sum(self, x: torch.Tensor):
        x_i32 = x.to(torch.int32)
        total = x_i32.sum()
        return total.clamp(self.qmin, self.qmax).to(torch.int16)

class FXP32Simulator:
    def __init__(self, frac_bits=16):
        self.total_bits = 32
        self.frac_bits = frac_bits
        self.scale = 2 ** frac_bits
        self.qmin = -2 ** (self.total_bits - 1)
        self.qmax = 2 ** (self.total_bits - 1) - 1

    def quantize(self, x_float):
        return (torch.round(x_float * self.scale)
                .clamp(self.qmin, self.qmax)
                .to(torch.int32))

    def dequantize(self, x_int):
        return x_int.to(torch.float32) / self.scale

    def add(self, x, y):
        return (x.to(torch.int64) + y.to(torch.int64)).clamp(self.qmin, self.qmax).to(torch.int32)

    def mul(self, x, y):
        prod = x.to(torch.int64) * y.to(torch.int64)
        return (prod >> self.frac_bits).clamp(self.qmin, self.qmax).to(torch.int32)

    def fxp_matmul(self, A, B):
        prod = torch.matmul(A.to(torch.int64), B.to(torch.int64))
        return (prod >> self.frac_bits).clamp(self.qmin, self.qmax).to(torch.int32)

    def fxp_einsum(self, equation, *operands):
        if equation == 'i,i->':
            x, y = operands
            prod = x.to(torch.int64) * y.to(torch.int64)
            summed = prod.sum()
            return (summed >> self.frac_bits).clamp(self.qmin, self.qmax).to(torch.int32)
        else:
            raise NotImplementedError(f"Equation '{equation}' not supported.")
        
    def fxp_conv1d(self, x: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0):
        """ FXP Conv1D 시뮬레이션 """
        x_i32 = x.to(torch.int64)
        w_i32 = weight.to(torch.int64)
        out = F.conv1d(x_i32, w_i32, bias=None, stride=stride, padding=padding)
        out = (out >> self.frac_bits).clamp(self.qmin, self.qmax).to(torch.int32)
        if bias is not None:
            out += bias.to(torch.int32).view(1, -1, 1)
        return out

    def fxp_conv2d(self, x: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0):
        """ FXP Conv2D 시뮬레이션 """
        x_i32 = x.to(torch.int64)
        w_i32 = weight.to(torch.int64)
        out = F.conv2d(x_i32, w_i32, bias=None, stride=stride, padding=padding)
        out = (out >> self.frac_bits).clamp(self.qmin, self.qmax).to(torch.int32)
        if bias is not None:
            out += bias.to(torch.int32).view(1, -1, 1, 1)
        return out
    
    def fxp_sum(self, x: torch.Tensor):
        x_i32 = x.to(torch.int64)
        total = x_i32.sum()
        return total.clamp(self.qmin, self.qmax).to(torch.int32)

if __name__ == '__main__':
    fxp = FXP16Simulator(frac_bits=14)
    fxp32 = FXP32Simulator()
    fxp8 = FXP8Simulator()

    # 기본 벡터
    a = torch.tensor([0.12345, 0.023456, -0.34567])
    b = torch.tensor([0.04556, -0.00578, -0.67893])

    # 양자화
    a_q = fxp32.quantize(a)
    b_q = fxp32.quantize(b)
    # print(a_q)
    # print(b_q)

    print("=== ADD ===")
    print("Float:", a + b)
    print("FXP  :", fxp32.dequantize(fxp32.add(a_q, b_q)))

    print("\n=== MUL ===")
    print("Float:", a * b)
    print("FXP  :", fxp32.dequantize(fxp32.mul(a_q, b_q)))

    print("\n=== DOT (einsum) ===")
    print("Float:", torch.dot(a, b))
    print("FXP  :", fxp32.dequantize(fxp32.fxp_einsum('i,i->', a_q, b_q)))

    print("\n=== MATMUL ===")
    A = torch.tensor([[0.12345, 0.023456], [0.34567, 0.00578]])
    B = torch.tensor([[0.04556, 0.67893], [-0.00071, 0.8]])
    A_q = fxp32.quantize(A)
    B_q = fxp32.quantize(B)
    print("Float:\n", torch.matmul(A, B))
    print("FXP:\n", fxp32.dequantize(fxp32.fxp_matmul(A_q, B_q)))

    # Conv1D 테스트
    x1 = torch.tensor([[[.12343, 0.023456, -0.34567]]])
    w1 = torch.tensor([[[0.00578, -0.04556, 0.67893]]])  # simple edge filter
    x1_q = fxp32.quantize(x1)
    w1_q = fxp32.quantize(w1)

    y1_true = F.conv1d(x1, w1, padding=1)
    y1_fxp = fxp32.dequantize(fxp32.fxp_conv1d(x1_q, w1_q, padding=1))

    print("=== Conv1D ===")
    print("Float:\n", y1_true)
    print("FXP  :\n", y1_fxp)
    print("Abs Error:\n", torch.abs(y1_true - y1_fxp))

    # Conv2D 테스트
    x2 = torch.tensor([[[[0.1214, 0.0043242, 0.4235433, 0.25325],
                        [0.2523, 0.04556, -0.00578, -0.67893],
                        [.12343, 0.023456, -0.34567, 0.00240],
                        [0.00578, -0.04556, 0.67893, -0.002]]]])
    w2 = torch.tensor([[[[0.1, 0.], [0., -0.1]]]])  # 2x2 difference filter

    x2_q = fxp32.quantize(x2)
    w2_q = fxp32.quantize(w2)

    y2_true = F.conv2d(x2, w2, padding=0)
    y2_fxp = fxp32.dequantize(fxp32.fxp_conv2d(x2_q, w2_q, padding=0))

    print("\n=== Conv2D ===")
    print("Float:\n", y2_true)
    print("FXP  :\n", y2_fxp)
    print("Abs Error:\n", torch.abs(y2_true - y2_fxp))

    print("\n=== RMS ===")
    x = torch.tensor([0.12345, 0.023456, -0.34567, 0.04556, -0.00578, -0.67893])
    # 양자화
    x_q = fxp32.quantize(x)
    # 1. 제곱 (x^2) = mul(x, x)
    x2_q = fxp32.mul(x_q, x_q)
    # 2. 누적합
    sum_q = x2_q.to(torch.int32).sum()  # int32 정수 합계
    # 3. 평균 = sum / D
    D = x.numel()
    mean_q = sum_q // D

    # 4. sqrt는 float에서 수행 (근사 가능)
    mean_f = fxp32.dequantize(mean_q)
    inv_sqrt = 1.0 / torch.sqrt(mean_f)

    print(f"sum(x^2)       = {sum_q}")
    print(f"mean(x^2)      = {mean_f:.6f}")
    print(f"scale = 1/sqrt(mean) = {inv_sqrt:.6f}")

    rms = 1.0 / torch.sqrt(((x*x).sum()/D))
    print(f'RMS:    = {rms}')