# 필요한 라이브러리 import
import time
import torch
from torch import nn
from transformers import AutoTokenizer
from mamba2 import Mamba2LMHeadModel, Mamba2Config, ssd, InferenceCache
import torch.nn.functional as F
from einops import rearrange

from FXP_simulator import FXP16Simulator, FXP32Simulator, FXP8Simulator
fxp16_14 = FXP16Simulator(frac_bits=14)
fxp16_12 = FXP16Simulator(frac_bits=12)
fxp16_11 = FXP16Simulator(frac_bits=11)
fxp16_8 =  FXP16Simulator(frac_bits=8)
fxp32 = FXP32Simulator(frac_bits=16)
fxp8_4 = FXP8Simulator(frac_bits=4)

from exp_graph import FastBiasedExp


from SiLU_graph import ApproxSiLU
approx = ApproxSiLU()

def findm(x):
    max_val = x.max()     # 최대값
    min_val = x.min()     # 최소값
    mean_val = x.mean()   # 평균값
    print(f"|{i}| Max: {max_val}, Min: {min_val}, Mean: {mean_val}")
    return 0


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



# CUDA가 가능하면 GPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mamba2 설정 정의: d_model=768, 24-layer, vocab_size=50277
config = Mamba2Config(d_model=768, n_layer=24, vocab_size=50277)

# 모델 초기화 및 양자화된 state_dict 로딩
model = Mamba2LMHeadModel(config)
model.load_state_dict(torch.load(r"C:\Internship\mamba2-130m\mamba2_130m_quantized.pth"))
model = model.to(device)

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
prompt = "I'm so happy!!"  # "The future of AI"
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
# RMS_Seg = SegmentedLUTRMSNorm(dim=config.d_model).to(device)  # 사용자 정의 RMS 생성
# RMS_Seg2 = SegmentedLUTRMSNorm(dim=1536).to(device)  # 사용자 정의 RMS 생성
# approx_silu = ApproxSiLU()
# # === LUT 정의 ===
# segment = torch.tensor([0.0, 0.0015625, 0.0125, 0.5, 1, 2, 64, 1024, 24576])
# esp = 1e-6
# # sq = 1 / (torch.sqrt(segment + esp))
# # LUT = [(sq[i] + sq[i + 1]) / 2 for i in range(len(sq) - 1)]
# LUT = [(1 / torch.sqrt((segment[i] + segment[i + 1]) / 2 + esp)) for i in range(len(segment) - 1)]
# LUT.append(LUT[-1])  # overflow 방지
# print(LUT)


# 자동 생성 반복: 최대 20 토큰 생성
with torch.no_grad():
    # for _ in range(1):  # 20
    for _ in range(40):  # 100
        seqlen = f_token.shape[1]  # 보통 1

        # f_token을 모델 입력 형식으로 변환
        # input_tensor = torch.tensor([[f_token]], device=device)
        input_tensor = f_token.to(device)

        # (1, 1) → 임베딩: (1, 1, d_model)
        u = model.backbone['embedding'](input_tensor)
        # print(u)
        u_p = u
        u = fxp16_12.quantize(u)
        u = fxp16_12.dequantize(u)
        # print(u)
        # print("Abs Error:\n", torch.sum(torch.abs(u - u_p)))
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

            # print(residual[0,0,:50], '\n\n')

            # # 1. residual → quantize → dequantize
            # temp = fxp16_14.quantize(residual)
            # x1 = fxp16_14.dequantize(temp)  # [B, L, D]

            # # 2. RMS^2 = mean(x^2)
            # x2 = x1.pow(2).mean(-1, keepdim=True) + esp  # shape: [B, L, 1]

            # # 3. LUT 보간 기반 역제곱근 근사
            # norm_scale = torch.full_like(x2, LUT[-1])  # 기본값은 마지막 LUT 값

            # for seg_idx in range(len(segment) - 1):
            #     s0 = segment[seg_idx]
            #     s1 = segment[seg_idx + 1]
            #     mask = (x2 >= s0) & (x2 < s1)  # bool mask

            #     y0 = 1.0 / torch.sqrt(s0 + esp)
            #     y1 = 1.0 / torch.sqrt(s1 + esp)

            #     # Linear interpolation
            #     interp = ((x2 - s0) * y1 + (s1 - x2) * y0) / (s1 - s0)
            #     norm_scale = torch.where(mask, interp, norm_scale)

            # # 4. 정규화 후 가중치 곱
            # gamma = model.backbone['layers'][i].norm.weight  # [D]
            # x = norm_scale * x1 * gamma

            # # 5. 비교 출력
            # print("Torch.rsqrt 결과:", torch.rsqrt(x1.pow(2).mean(-1, keepdim=True) + esp)[0, 0, :10])
            # print("LUT 근사 결과:", norm_scale[0, 0, :10])
            # print("출력 x:", x[0, 0, :50])

# '''
            # Linear projection → z, xBC, dt 분리
            zxbcdt = model.backbone['layers'][i]['mixer'].in_proj(x.squeeze(1))
            z, xBC, dt = torch.split(
                zxbcdt,
                [config.d_inner, config.d_inner + 2 * config.d_state, config.nheads],
                dim=-1,
            )

            xBC_p = xBC
            xBC = fxp16_11.quantize(xBC)
            xBC = fxp16_11.dequantize(xBC)
            print(f"y {i} Abs Error:", torch.sum(torch.abs(xBC - xBC_p)))



            # convolution을 위한 상태 이동 및 입력 추가
            h[i].conv_state.copy_(torch.roll(h[i].conv_state, shifts=-1, dims=-1))
            h[i].conv_state[:, :, -1] = xBC

            # 1D depthwise convolution (수동 구현)
            xBC = torch.sum(
                h[i].conv_state * rearrange(model.backbone['layers'][i]['mixer'].conv1d.weight, "d 1 w -> d w"),
                dim=-1
            )

            xBC += model.backbone['layers'][i]['mixer'].conv1d.bias

            # xBC = F.silu(xBC)
            xBC = approx(xBC)  # 근사 SiLU 사용


            # x, B, C 분리
            x, B, C = torch.split(xBC, [config.d_inner, config.d_state, config.d_state], dim=-1)

            # A = -torch.exp(model.backbone['layers'][i]['mixer'].A_log)  # state decay factor
            A = -FastBiasedExp()(model.backbone['layers'][i]['mixer'].A_log)

            # SSM 계산
            
            dt = F.softplus(dt + model.backbone['layers'][i]['mixer'].dt_bias)
            # print('dt: ', dt)
            dt_p = dt
            dt = fxp16_12.quantize(dt)
            dt = fxp16_12.dequantize(dt)
            # print(f"dt {i} Abs Error:", torch.sum(torch.abs(dt - dt_p)))
            # dA = torch.exp(dt * A)
            dA = FastBiasedExp()(dt * A)
            # print('dA:  ', dA)
            dA_p = dA
            dA = fxp16_12.quantize(dA)
            dA = fxp16_12.dequantize(dA)
            # print(f"dA {i} Abs Error:", torch.sum(torch.abs(dA - dA_p)))
            x = rearrange(x, "b (h p) -> b h p", p=config.headdim)

            # 상태 업데이트
            dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
            h[i].ssm_state.copy_(h[i].ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)

            # 출력 계산
            y = torch.einsum("bhpn, bn -> bhp", h[i].ssm_state, C)
            y = y + rearrange(model.backbone['layers'][i]['mixer'].D, "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            # print('y: ', y)
            # y_p = y
            # y = fxp16_12.quantize(y)  # error 너무 크다.
            # y = fxp16_12.dequantize(y)
            # print(f"y {i} Abs Error:", torch.sum(torch.abs(y - y_p)))
            # print('z: ', z)
            z_p = z
            if i <= 9:
                z = fxp16_11.quantize(z)  # fxp8_4
                z = fxp16_11.dequantize(z)
            else:
                z = fxp16_12.quantize(z)  # fxp8_4
                z = fxp16_12.dequantize(z)
            # print(f"z {i} Abs Error:", torch.sum(torch.abs(z - z_p)))

            z = approx(z)
            z = fxp16_14.quantize(z)
            z = fxp16_14.dequantize(z)
            y = y * z
            # y = model.backbone['layers'][i]['mixer'].norm(y, z)
            y = model.backbone['layers'][i]['mixer'].norm(y)

            # y = RMS_Seg2(y * F.silu(z), weight = model.backbone['layers'][i]['mixer'].norm.weight)
            # y = RMS_Seg2(y * approx_silu(z), weight = model.backbone['layers'][i]['mixer'].norm.weight)
            
            y = model.backbone['layers'][i]['mixer'].out_proj(y)
            
            # residual connection
            residual = residual + y.unsqueeze(1)



            # max_val = residual.max().item()
            # min_val = residual.min().item()

            # print(f"{i}  최댓값:", max_val, "최솟값:", min_val)

            
            residual_p = residual
            residual_q = fxp32.quantize(residual_p)
            residual_dq = fxp32.dequantize(residual_q)
            residual = residual_dq

            # 오차 디버깅 라인
            # # 오차 계산
            # abs_error = torch.abs(residual_dq - residual_p)
            # threshold = 1
            # # 조건: 오차가 threshold보다 큰 요소
            # mask = abs_error > threshold
            # if mask.any():
            #     print(f"\n⚠️ [Layer {i}] Abs Error: {abs_error.sum().item():.6f} | Elements over {threshold}: {mask.sum().item()}개")

            #     indices = torch.nonzero(mask, as_tuple=False)
            #     for idx in indices:
            #         # idx는 텐서 → 리스트 변환 후 unpack
            #         idx_tuple = tuple(idx.tolist())

            #         # 동적으로 인덱싱
            #         orig_val = residual_p[idx_tuple].item()
            #         deq_val = residual_dq[idx_tuple].item()
            #         error_val = abs_error[idx_tuple].item()

            #         print(f" → index {idx_tuple} | original: {orig_val:.6f}, dequantized: {deq_val:.6f}, abs error: {error_val:.6f}")
            # else:
            #     print(f"[Layer {i}] 모든 요소 오차 < {threshold}")


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
        # f_token = next_token.unsqueeze(0)
        f_token = next_token.to(device).unsqueeze(0)

# 토큰 결과 디코딩 후 출력
print(tokenizer.decode(generated, skip_special_tokens=True))
# '''