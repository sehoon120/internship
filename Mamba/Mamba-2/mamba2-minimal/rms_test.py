# 필요한 라이브러리 import
import time
import torch
from torch import nn
from transformers import AutoTokenizer
from mamba2 import Mamba2LMHeadModel, Mamba2Config, ssd, InferenceCache
import torch.nn.functional as F
from einops import rearrange

def quant_dequant_fxpa(tensor: torch.Tensor, total_bits: int = 8, frac_bits: int = 6):
    """
    Quantize + dequantize a tensor using signed fixed-point FXPa format.
    
    Parameters:
    - tensor: input float32 tensor
    - total_bits: total bit width (e.g. 8, 6, etc.)
    - frac_bits: fractional bit count (e.g. 4 for Q3.4)
    
    Returns:
    - float32 tensor with FXPa precision
    """
    assert total_bits >= 2, "total_bits must be >= 2 (1 sign bit + at least 1 data bit)"
    int_bits = total_bits - 1 - frac_bits
    scale = 2 ** frac_bits

    qmin = -2 ** (total_bits - 1)
    qmax = 2 ** (total_bits - 1) - 1

    # 1. Quantization: float → fixed-point int
    q_tensor = torch.round(tensor * scale).clamp(qmin, qmax).to(torch.int32)

    # 2. Dequantization: fixed-point int → float
    dq_tensor = q_tensor.to(torch.float32) / scale

    return dq_tensor

# hfqmamba식 RMS
class SegmentedLUTRMSNorm(nn.Module):
    def __init__(self, dim: int, segments=None, dtype=torch.float32, eps=1e-6, device=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.device = device
        self.dtype = dtype

        # 세그먼트 정의
        # 8 4 2 2 32 16 24
        self.segments = segments or [0.0, 0.0015625, 0.0125, 0.5, 1, 2, 64, 1024, 24576]
        self.segments = torch.tensor(self.segments, dtype=torch.float32, device=device)

        # 중심 기반 1/sqrt(x) LUT 계산
        self.lut = self._generate_inv_sqrt_lut(self.segments, dtype)

    def _generate_inv_sqrt_lut(self, segments, dtype=torch.float32):
        lut_values = []
        for i in range(len(segments) - 1):
            # mid = (segments[i] + segments[i+1]) / 2
            mid = (segments[i] + (segments[i+1]-segments[i])/3.5)
            if mid <= 0:
                lut_values.append(0.0)
            else:
                lut_values.append(1.0 / torch.sqrt(mid))
        return torch.tensor(lut_values, dtype=dtype, device=self.device)

    def _lut_lookup(self, mean_sq):
        """
        mean_sq: Tensor of shape (B,) or (B, 1), float32
        """
        mean_sq = mean_sq.view(-1)
        out = torch.zeros_like(mean_sq)

        for i in range(len(self.segments) - 1):
            lower = self.segments[i]
            upper = self.segments[i+1]
            mask = (mean_sq >= lower) & (mean_sq < upper)
            out[mask] = self.lut[i]

        out[mean_sq >= self.segments[-1]] = self.lut[-1]
        return out.view(-1, 1)  # shape: (B, 1)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) or (B, 1, D) float32 tensor
        weight: (D,) float32 tensor
        """
        original_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])  # (B*L, D)

        # 1. mean of squares
        mean_sq = x.pow(2).mean(dim=-1)  # shape: (B,)

        # 2. LUT 기반 1/√mean_sq 근사
        inv_rms = self._lut_lookup(mean_sq)  # shape: (B, 1)

        # 3. 정규화 + weight 적용
        x_norm = x * inv_rms  # broadcast (B, D)
        x_out = x_norm * weight  # apply weight

        return x_out.view(original_shape)

# hfqmamba + 선형 보간
class SegmentedLUTRMSNorm_linear_ver(nn.Module):
    def __init__(self, dim: int, segments=None, dtype=torch.float32, eps=1e-6, device=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.device = device
        self.dtype = dtype

        # 세그먼트 정의 (예시 구간)
        self.segments = segments or [0.0, 0.0015625, 0.0125, 0.5, 1, 2, 64, 1024, 24576]
        self.segments = torch.tensor(self.segments, dtype=torch.float32, device=device)

        # LUT 값 생성: 각 구간의 중심값 또는 맞춤 midpoint
        self.lut = self._generate_inv_sqrt_lut(self.segments, dtype)

    def _generate_inv_sqrt_lut(self, segments, dtype=torch.float32):
        lut_values = []
        for i in range(len(segments)):
            val = 1.0 / torch.sqrt(segments[i] + 0.0001)
            lut_values.append(val)

        return torch.tensor(lut_values, dtype=dtype, device=self.device)


    def _interp_lut_lookup(self, mean_sq):
        """
        선형 보간 기반 1/√mean_sq 근사
        Input: mean_sq (B,) float32
        Output: inv_rms (B, 1) float32
        """
        mean_sq = mean_sq.view(-1)  # (B,)
        out = torch.zeros_like(mean_sq)

        for i in range(len(self.segments) - 1):
            lower = self.segments[i]
            upper = self.segments[i + 1]
            mask = (mean_sq >= lower) & (mean_sq < upper)

            if mask.any():
                # 보간 비율 계산
                ratio = (mean_sq[mask] - lower) / (upper - lower + self.eps)
                lut_l = self.lut[i]
                lut_h = self.lut[i + 1]
                out[mask] = lut_l * (1 - ratio) + lut_h * ratio

        # 마지막 구간 이상 처리
        out[mean_sq >= self.segments[-1]] = self.lut[-1]
        return out.view(-1, 1)  # (B, 1)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) or (B, 1, D) float32
        weight: (D,) float32
        """
        original_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])  # (B*L, D)

        # 1. mean of squares
        mean_sq = x.pow(2).mean(dim=-1)  # (B,)

        # 2. LUT + 선형 보간 기반 inv sqrt 근사
        inv_rms = self._interp_lut_lookup(mean_sq)  # (B, 1)

        # 3. 정규화 + weight
        x_norm = x * inv_rms  # (B, D)
        x_out = x_norm * weight  # broadcasting (D,)

        return x_out.view(original_shape)

# RMS는 건들면 정확도가 너무 흔들린다..

# SiLU 선형 근사 
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

# CUDA가 가능하면 GPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mamba2 설정 정의: d_model=768, 24-layer, vocab_size=50277
config = Mamba2Config(d_model=768, n_layer=24, vocab_size=50277)

# 모델 초기화 및 양자화된 state_dict 로딩
model = Mamba2LMHeadModel(config)
model.load_state_dict(torch.load(r"C:\Internship\mamba2-130m\mamba2_130m_quantized.pth"))

# GPT-NeoX 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id

# 레이어 수만큼 inference cache (conv_state, ssm_state) 초기화
h = [InferenceCache.alloc(
    batch_size=1,
    args=config,
    device=device
) for _ in range(config.n_layer)]

# 입력 프롬프트 정의 및 토크나이징
prompt = "The future of AI"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)  # shape: (1, L)

# 마지막 토큰을 f_token으로 분리, 나머지는 prefix
prefix, f_token = input_ids[:, :-1], input_ids[:, -1:]

# 모델 내부 chunk size를 기준으로 처리 가능한 구간 계산
chunk_size = model.args.chunk_size
n_chunked = (prefix.shape[1] // chunk_size) * chunk_size

# chunk 단위는 한번에 처리하고, 나머지는 step-by-step으로 처리
if n_chunked > 0:
    _, h = model(prefix[:, :n_chunked], None)
else:
    h = [InferenceCache.alloc(1, model.args, device=device) for _ in range(model.args.n_layer)]

# 남은 prefix 토큰들 처리 (inference step 방식)
for i in range(n_chunked, prefix.shape[1]):
    _, h = model(prefix[:, i:i+1], h)

# x = torch.tensor([0.1234, -0.9876, 3.14, -2.71])
# x_fxp16 = quant_dequant_fxpa(x, total_bits=16, frac_bits=12)
# print(x_fxp16)



# 생성된 결과 토큰 저장 리스트 초기화
generated = [t.item() for t in input_ids[0]]
RMS_Seg = SegmentedLUTRMSNorm_linear_ver(dim=config.d_model).to(device)  # 사용자 정의 RMS 생성
RMS_Seg2 = SegmentedLUTRMSNorm_linear_ver(dim=1536).to(device)  # 사용자 정의 RMS 생성
approx_silu = ApproxSiLU()
# 자동 생성 반복: 최대 20 토큰 생성
with torch.no_grad():
    # for _ in range(1):  # 20
    for _ in range(30):
        seqlen = f_token.shape[1]  # 보통 1

        # f_token을 모델 입력 형식으로 변환
        input_tensor = torch.tensor([[f_token]], device=device)

        # (1, 1) → 임베딩: (1, 1, d_model)
        u = model.backbone['embedding'](input_tensor)
        residual = u  # skip connection용

        # 전체 레이어 순차 처리
        # for i in range(1):  # config.n_layer
        for i in range(config.n_layer):
            # residual = torch([[[0, 00.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]]])
            x = model.backbone['layers'][i].norm(residual)  # RMSNorm
            # print(x[0,0,:50], '\n\n')
            # x = quant_dequant_fxpa(x, total_bits=16, frac_bits=14)  # Intermediate quantizaiton
            # print(x[0,0,:50], '\n\n')
            # x = RMS_Seg(residual, weight = model.backbone['layers'][i].norm.weight)
            # print(x[0,0,:30], '\n\n')

# '''
            # Linear projection → z, xBC, dt 분리
            zxbcdt = model.backbone['layers'][i]['mixer'].in_proj(x.squeeze(1))
            z, xBC, dt = torch.split(
                zxbcdt,
                [config.d_inner, config.d_inner + 2 * config.d_state, config.nheads],
                dim=-1,
            )
            # print(z[0,:50], '\n\n')
            # print(xBC[0,:50], '\n\n')
            # print(dt[0,:50], '\n\n')
            # z = quant_dequant_fxpa(z)
            # xBC = quant_dequant_fxpa(xBC)
            # dt = quant_dequant_fxpa(dt)
            # print(z[0,:50], '\n\n')
            # print(xBC[0,:50], '\n\n')
            # print(dt[0,:50], '\n\n')

            # convolution을 위한 상태 이동 및 입력 추가
            h[i].conv_state.copy_(torch.roll(h[i].conv_state, shifts=-1, dims=-1))
            h[i].conv_state[:, :, -1] = xBC

            # 1D depthwise convolution (수동 구현)
            xBC = torch.sum(
                h[i].conv_state * rearrange(model.backbone['layers'][i]['mixer'].conv1d.weight, "d 1 w -> d w"),
                dim=-1
            )
            # xBC = quant_dequant_fxpa(xBC, total_bits=16, frac_bits=14)
            xBC += model.backbone['layers'][i]['mixer'].conv1d.bias
            # xBC = quant_dequant_fxpa(xBC, total_bits=16, frac_bits=14)
            xBC = F.silu(xBC)
            # xBC = approx_silu(xBC)
            # xBC = quant_dequant_fxpa(xBC, total_bits=16, frac_bits=14)

            # x, B, C 분리
            x, B, C = torch.split(xBC, [config.d_inner, config.d_state, config.d_state], dim=-1)

            A = -torch.exp(model.backbone['layers'][i]['mixer'].A_log)  # state decay factor

            # SSM 계산
            dt = F.softplus(dt + model.backbone['layers'][i]['mixer'].dt_bias)
            dA = torch.exp(dt * A)
            x = rearrange(x, "b (h p) -> b h p", p=config.headdim)

            # 상태 업데이트
            dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
            h[i].ssm_state.copy_(h[i].ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)

            # 출력 계산
            y = torch.einsum("bhpn, bn -> bhp", h[i].ssm_state, C)
            y = y + rearrange(model.backbone['layers'][i]['mixer'].D, "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")

            # y = model.backbone['layers'][i]['mixer'].norm(y, z)

            y = RMS_Seg2(y * F.silu(z), weight = model.backbone['layers'][i]['mixer'].norm.weight)
            # y = RMS_Seg2(y * approx_silu(z), weight = model.backbone['layers'][i]['mixer'].norm.weight)
            
            y = model.backbone['layers'][i]['mixer'].out_proj(y)

            # residual connection
            residual = residual + y.unsqueeze(1)

        # 최종 layer norm 후 LM 헤드
        residual = model.backbone.norm_f(residual)
        logits = model.lm_head(residual)  # shape: (1, 1, vocab_size)
        out = logits[:, :seqlen]  # seqlen=1
        logits = out[0, -1]  # 최종 토큰의 로짓

        # 다음 토큰 샘플링
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 종료 조건 확인 (EOS 토큰일 경우)
        if next_token.item() == tokenizer.eos_token_id:
            break

        # 다음 루프 준비
        generated.append(next_token.item())
        f_token = next_token.unsqueeze(0)

# 토큰 결과 디코딩 후 출력
print(tokenizer.decode(generated, skip_special_tokens=True))
# '''