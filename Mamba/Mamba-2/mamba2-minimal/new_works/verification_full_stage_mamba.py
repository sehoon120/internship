import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import math


def exp_fast8(x: torch.Tensor) -> torch.Tensor:
    """
    8-segment PWL 근사로 e^x 계산.
    e^x = 2^(x*log2(e)) = 2^(k+f) = 2^k * 2^f
    2^f는 f∈[0,1)에서 8구간(pwl) 1차 근사.
    """
    # 상수/준비
    log2e = 1.4423828  # 1.4426950408889634  # 1/ln(2)
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

def save_tensor_as_hex(tensor, path):
    tensor = tensor.to(torch.float16).contiguous()
    with open(path, 'w') as f:
        for val in tensor.view(-1):
            u16 = val.view(torch.uint16).item()
            f.write(f"{u16:04x}\n")

def load_hex_tensor(filepath, shape):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = [int(line.strip(), 16) for line in lines]
    arr = np.array(data, dtype=np.uint16).view(np.float16)
    return torch.tensor(arr, dtype=torch.float16).reshape(shape)

def print_tensor_fp16_hex_inline(tensor):
    tensor = tensor.to(torch.float16).contiguous()
    flat = tensor.view(-1)
    for val in flat:
        u16 = val.view(torch.uint16).item()
        print(f"{u16:04x}", end=' ')  # 줄바꿈 없이 공백 구분
    print('\n')  # 마지막 줄바꿈


B_ = 1
H_ = 24
P_ = 64
N_ = 128

M_ = int(H_*P_/2)  # 768
C_ = H_*P_+2*N_  # 1792
T_ = H_*P_+C_+H_  # 3352

h_slice = 1
p_slice = 1
n_slice = 128
m_slice = 32
c_slice = 128

eps = 1e-5

# 경로 지정
base_path = "C:/Internship/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_full_mamba_datas"  # "C:/Internship/intermediate_datas"
c_b = load_hex_tensor(f"{base_path}/0_c_b.hex", (C_))
c_s = load_hex_tensor(f"{base_path}/0_c_s.hex", (B_, C_, 4))
c_W = load_hex_tensor(f"{base_path}/0_c_W.hex", (C_, 4))
in_proj_W = load_hex_tensor(f"{base_path}/0_in_proj_W.hex", (T_, M_))  # 2574336/
out_proj_W = load_hex_tensor(f"{base_path}/0_out_proj_W.hex", (M_, H_*P_))
RMS_W1 = load_hex_tensor(f"{base_path}/0_RMS_W1.hex", (M_))
RMS_W2 = load_hex_tensor(f"{base_path}/0_RMS_W2.hex", (H_*P_))
dt_bias = load_hex_tensor(f"{base_path}/0_dt_bias_w.hex", (B_, H_))
A = load_hex_tensor(f"{base_path}/0_A.hex", (H_,))
D = load_hex_tensor(f"{base_path}/0_D.hex", (H_,))
h_prev = load_hex_tensor(f"{base_path}/0_h_ssm_state.hex", (B_, H_, P_, N_))

x_inter = load_hex_tensor(f"{base_path}/0_x.hex", (B_, H_, P_))  # SSM start
dt_inter = load_hex_tensor(f"{base_path}/0_dt.hex", (B_, H_))  # inter
B_inter = load_hex_tensor(f"{base_path}/0_B.hex", (B_, N_))  # inter
C_inter = load_hex_tensor(f"{base_path}/0_C.hex", (B_, N_))  # inter
y_SSM_inter = load_hex_tensor(f"{base_path}/0_y_SSM.hex", (B_, H_, P_))  # SSM end

residual = load_hex_tensor(f"{base_path}/0_residual.hex", (B_, 1, M_))   # start
y_out = load_hex_tensor(f"{base_path}/0_y.hex", (B_, 1, M_))  # out

Y_SSM = torch.zeros((B_, H_, P_), dtype=torch.float16)
ln2 = math.log(2)


# Mamba start
u = residual  # b 1 m
# RMS norm    
u_2 = u*u  # b 1 m
u_s = u_2.sum(dim=-1)  # b 1
u_m = u_s/M_ + eps  # b 1
u_r = u_m.sqrt()  # b 1
u_rms = u/u_r  # b 1 m
u_norm = u_rms*RMS_W1  # b 1 m
# in_proj
in_p = torch.einsum('blm,tm->bmt', u_norm, in_proj_W)  # b m t
in_p_s = in_p.sum(-2)  # b t
# split
z, xBC, dt = torch.split(
    in_p_s,
    [H_*P_, C_, H_],
    dim=-1,
)
# conv
c_s = torch.roll(c_s, shifts=-1, dims=-1)  # b c 4
c_s[:, :, -1] = xBC  # b c 4
a = c_s * c_W  # b c 4
xBC = a.sum(dim=-1)  # b c
xBC += c_b  # b c
# silu
xBC_s = torch.exp(-xBC)  # b c
mask = (-xBC) > 11.0   # overflow 검사
if mask.any().item():
    max_val = (-xBC)[mask].max().item()
    print(f"[Warning] exp(-xBC) overflow risk! max(-xBC) = {max_val:.3f}")
xBC_s += 1  # b c
xBC_s = 1/xBC_s  # b c
xBC = xBC*xBC_s  # b c
# split
x, B, C = torch.split(
    xBC, 
    [H_*P_, N_, N_], 
    dim=-1
)
x = rearrange(x, "b (h p) -> b h p", p=P_)  # b h p
# 결과 저장
save_tensor_as_hex(x, f"{base_path}/0_x_python.hex")
# SSM_tiled
for h_idx in range(0, H_, h_slice):
    dt_bias_tile = dt_bias[:, h_idx:h_idx+h_slice]
    dt_tile = dt[:, h_idx:h_idx+h_slice]
    A_tile = A[h_idx:h_idx+h_slice]
    D_tile = D[h_idx:h_idx+h_slice]
    for p_idx in range(0, P_, p_slice):
        x_tile = x[:, h_idx:h_idx+h_slice, p_idx:p_idx+p_slice]
        for n_idx in range(0, N_, n_slice):
            B_tile = B[:, n_idx:n_idx+n_slice]
            C_tile = C[:, n_idx:n_idx+n_slice]
            h_tile = h_prev[:, h_idx:h_idx+h_slice, p_idx:p_idx+p_slice, n_idx:n_idx+n_slice]
            dt_sp_tile = F.softplus(dt_tile + dt_bias_tile)
            dA_tile = torch.exp(dt_sp_tile * A_tile)
            dx_tile = torch.einsum("bh, bhp -> bhp", dt_sp_tile, x_tile)
            dxB_tile = torch.einsum("bhp, bn -> bhpn", dx_tile, B_tile)
            h_new = h_tile * rearrange(dA_tile, "b h -> b h 1 1") + dxB_tile
            y = torch.einsum("bhpn, bn -> bhp", h_new, C_tile)
            y = y + rearrange(D_tile, "h -> h 1") * x_tile
        Y_SSM[:, h_idx:h_idx+h_slice, p_idx:p_idx+p_slice] += y
Y_SSM = rearrange(Y_SSM, "b h p -> b (h p)")  # b hp
# 결과 저장
save_tensor_as_hex(Y_SSM, f"{base_path}/0_y_SSM_python.hex")
# silu
z_s = torch.exp(-z)  # hp
mask = (-z) > 11.0   # overflow 검사
if mask.any().item():
    max_val = (-z)[mask].max().item()
    print(f"[Warning] exp(-z) overflow risk! max(-z) = {max_val:.3f}")
z_s += 1  # hp
z_s = 1/z_s  # hp
z = z*z_s  # hp
# y*z
y = Y_SSM*z  # b hp
# RMS norm
y_2 = y*y  # b hp
y_s = y_2.sum(dim=-1)  # b
y_m = y_s/(H_*P_) + eps  # b
y_r = y_m.sqrt()  # b
y_rms = y/y_r.unsqueeze(-1)  # b hp
y_norm = y_rms*RMS_W2  # b hp
# out_proj
out_p = torch.einsum('ba,ma->bma', y_norm, out_proj_W)  # b m hp
out_p_s = out_p.sum(-1)  # b m
y_out = out_p_s.unsqueeze(1) + residual  # b 1 m


# 여기까지 완료 - 검증은 X


def load_fp16_hex(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = [int(line.strip(), 16) for line in lines]
    return torch.tensor(np.array(data, dtype=np.uint16).view(np.float16), dtype=torch.float16)

def compare_fp16_hex(file1, file2):
    t1 = load_fp16_hex(file1)
    t2 = load_fp16_hex(file2)

    if t1.shape != t2.shape:
        raise ValueError(f"파일 길이가 다릅니다: {t1.shape} vs {t2.shape}")

    abs_diff = torch.abs(t1 - t2)
    max_error = abs_diff.max().item()
    mean_error = abs_diff.mean().item()
    rms_error = torch.sqrt(torch.mean((t1 - t2) ** 2)).item()
    num_errors = (abs_diff > 1e-3).sum().item()  # 0.001 이상 차이 나는 항목 수

    print(f"🔍 Total elements: {t1.numel()}")
    print(f"📊 Max abs error : {max_error}")
    print(f"📊 Mean abs error: {mean_error}")
    print(f"📐 RMS error: {rms_error}")
    print(f"⚠️  Elements with >1e-3 error: {num_errors}")

    # 차이가 나는 인덱스 출력 (상위 10개)
    topk = torch.topk(abs_diff, k=10)
    print("\nTop 10 max error entries:")
    for idx, val in zip(topk.indices, topk.values):
        i = idx.item()
        print(f"[{i}] Py={t1[i].item():.6f}, Verilog={t2[i].item():.6f}, AbsErr={val.item():.6f}")

    # # ➕ 추가: t1에서 절대값이 2 또는 3보다 큰 값 찾기
    # mask2 = torch.abs(t1) > 2.0
    # mask3 = torch.abs(t1) > 3.0
    # idx2 = torch.nonzero(mask2, as_tuple=False).flatten().tolist()
    # idx3 = torch.nonzero(mask3, as_tuple=False).flatten().tolist()

    # print(f"\n🔎 |t1| > 2.0 → {len(idx2)} elements")
    # if idx2:
    #     print("  indices:", idx2[:20], "..." if len(idx2) > 20 else "")
    # print(f"🔎 |t1| > 3.0 → {len(idx3)} elements")
    # if idx3:
    #     print("  indices:", idx3[:20], "..." if len(idx3) > 20 else "")


# 사용 예시
file_pth = "C:/Internship/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_full_mamba_datas"
file_py = f"{file_pth}/0_y_SSM_python.hex"  # 0_y_out_python.hex"
file_v =  f"{file_pth}/0_y_SSM.hex"  # <- verilog 결과
compare_fp16_hex(file_py, file_v)