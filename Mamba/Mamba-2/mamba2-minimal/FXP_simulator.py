import torch

class FXP16Simulator:
    def __init__(self, frac_bits=14):
        self.total_bits = 16
        self.frac_bits = frac_bits
        self.scale = 2 ** frac_bits
        self.qmin = -2 ** (self.total_bits - 1)
        self.qmax =  2 ** (self.total_bits - 1) - 1

    def quantize(self, x_float: torch.Tensor):
        """ float32 → int16 FXP16 """
        x_scaled = torch.round(x_float * self.scale)
        x_clamped = x_scaled.clamp(self.qmin, self.qmax)
        return x_clamped.to(torch.int16)

    def dequantize(self, x_int: torch.Tensor):
        """ int16 FXP16 → float32 """
        return x_int.to(torch.float32) / self.scale

    def add(self, x: torch.Tensor, y: torch.Tensor):
        """ 정수 덧셈 + 클램핑 """
        z = x.to(torch.int32) + y.to(torch.int32)
        return z.clamp(self.qmin, self.qmax).to(torch.int16)

    def mul(self, x: torch.Tensor, y: torch.Tensor):
        """ 정수 곱셈 후 시프트 정규화 + 클램핑 """
        prod = x.to(torch.int32) * y.to(torch.int32)
        z = (prod >> self.frac_bits).clamp(self.qmin, self.qmax)
        return z.to(torch.int16)

    def fxp_matmul(self, A: torch.Tensor, B: torch.Tensor):
        """ FXP16 행렬 곱셈: (M×K) @ (K×N) = M×N """
        A_int = A.to(torch.int32)
        B_int = B.to(torch.int32)
        Z = torch.matmul(A_int, B_int) >> self.frac_bits
        return Z.clamp(self.qmin, self.qmax).to(torch.int16)


if(__name__ == '__main__'):
    fxp = FXP16Simulator(frac_bits=14)

    # 예시 벡터
    a = torch.tensor([-0.2812, 0.4531])
    b = torch.tensor([0.0469, -0.1875])

    # 양자화
    a_q = fxp.quantize(a)
    b_q = fxp.quantize(b)

    # 곱셈
    z_q = fxp.mul(a_q, b_q)
    z_float = fxp.dequantize(z_q)

    print(f"Input a (float): {a}")
    print(f"Input b (float): {b}")
    print(f"a * b (FXP result, float): {z_float}")
