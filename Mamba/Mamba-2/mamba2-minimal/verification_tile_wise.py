import torch
import numpy as np
from einops import rearrange

def _read_hex_lines(filepath):
    with open(filepath, 'r') as f:
        # 공백/빈 줄 제거
        return [ln.strip() for ln in f if ln.strip()]
    
def save_tensor_as_hex(tensor, path):
    tensor = tensor.to(torch.float16).contiguous()
    with open(path, 'w') as f:
        for val in tensor.view(-1):
            u16 = val.view(torch.uint16).item()
            f.write(f"{u16:04x}\n")

def load_fp32_hex(filepath: str, shape=None, device=None, strict: bool = True) -> torch.Tensor:
    """
    1줄=8hex(32bit) FP32 파일 로드.
    shape가 주어지면 reshape, strict면 원소수 검증.
    """
    lines = _read_hex_lines(filepath)
    data = [int(x, 16) for x in lines]
    arr = np.array(data, dtype=np.uint32).view(np.float32)
    t = torch.from_numpy(arr).to(torch.float32)
    if shape is not None:
        if strict and t.numel() != int(np.prod(shape)):
            raise ValueError(f"원소 수 불일치: file={t.numel()} vs shape={np.prod(shape)}")
        t = t.reshape(shape)
    if device is not None:
        t = t.to(device)
    return t

# def load_hex_tensor(filepath, shape):  # fp16 읽기
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = [int(line.strip(), 16) for line in lines]
    arr = np.array(data, dtype=np.uint16).view(np.float16)
    return torch.tensor(arr, dtype=torch.float16).reshape(shape)

def load_fp16_hex(filepath: str, shape=None, device=None, strict: bool = True) -> torch.Tensor:
    """
    1줄=4hex(16bit) FP16 파일 로드.
    shape가 주어지면 reshape, strict면 원소수 검증.
    """
    lines = _read_hex_lines(filepath)
    data = [int(x, 16) for x in lines]
    arr = np.array(data, dtype=np.uint16).view(np.float16)
    t = torch.from_numpy(arr).to(torch.float16)
    if shape is not None:
        if strict and t.numel() != int(np.prod(shape)):
            raise ValueError(f"원소 수 불일치: file={t.numel()} vs shape={np.prod(shape)}")
        t = t.reshape(shape)
    if device is not None:
        t = t.to(device)
    return t


def print_tensor_fp16_hex_inline(tensor):
    tensor = tensor.to(torch.float16).contiguous()
    flat = tensor.view(-1)
    for val in flat:
        u16 = val.view(torch.uint16).item()
        print(f"{u16:04x}", end=' ')  # 줄바꿈 없이 공백 구분
    print('\n')  # 마지막 줄바꿈


# B_ = 1
# H_ = 24
# P_ = 4
# N_ = 32

B_ = 1
H_ = 24
P_ = 64
N_ = 128

h_slice = 12
p_slice = 16
n_slice = 128  # 128  # 이 축으로는 slice 불가

# 경로 지정
base_path = "C:/Internship/intermediate_datas"
dA = load_fp32_hex(f"{base_path}/0_dA_fp32.hex", (B_, H_))
dt = load_fp32_hex(f"{base_path}/0_dt_fp32.hex", (B_, H_))
x = load_fp32_hex(f"{base_path}/0_x_fp32.hex", (B_, H_, P_))
B = load_fp32_hex(f"{base_path}/0_B_fp32.hex", (B_, N_))
C = load_fp32_hex(f"{base_path}/0_C_fp32.hex", (B_, N_))
D = load_fp32_hex(f"{base_path}/0_D_fp32.hex", (H_,))
h_prev = load_fp32_hex(f"{base_path}/0_ssm_state_fp32.hex", (B_, H_, P_, N_))

# fp16으로도 저장하기
save_tensor_as_hex(dA, f"{base_path}/0_dA_fp16.hex")
save_tensor_as_hex(dt, f"{base_path}/0_dt_fp16.hex")
save_tensor_as_hex(x, f"{base_path}/0_x_fp16.hex")
save_tensor_as_hex(B, f"{base_path}/0_B_fp16.hex")
save_tensor_as_hex(C, f"{base_path}/0_C_fp16.hex")
save_tensor_as_hex(D, f"{base_path}/0_D_fp16.hex")
save_tensor_as_hex(h_prev, f"{base_path}/0_ssm_state_fp16.hex")

# fp16으로 불러오기
dA = load_fp16_hex(f"{base_path}/0_dA_fp16.hex", (B_, H_))
dt = load_fp16_hex(f"{base_path}/0_dt_fp16.hex", (B_, H_))
x = load_fp16_hex(f"{base_path}/0_x_fp16.hex", (B_, H_, P_))
B = load_fp16_hex(f"{base_path}/0_B_fp16.hex", (B_, N_))
C = load_fp16_hex(f"{base_path}/0_C_fp16.hex", (B_, N_))
D = load_fp16_hex(f"{base_path}/0_D_fp16.hex", (H_,))
h_prev = load_fp16_hex(f"{base_path}/0_ssm_state_fp16.hex", (B_, H_, P_, N_))

Y = torch.zeros((B_, H_, P_), dtype=torch.float16)

for h_idx in range(0, H_, h_slice):
    for p_idx in range(0, P_, p_slice):
        for n_idx in range(0, N_, n_slice):
            dA_tile = dA[:, h_idx:h_idx+h_slice]
            dt_tile = dt[:, h_idx:h_idx+h_slice]
            x_tile = x[:, h_idx:h_idx+h_slice, p_idx:p_idx+p_slice]
            B_tile = B[:, n_idx:n_idx+n_slice]
            C_tile = C[:, n_idx:n_idx+n_slice]
            D_tile = D[h_idx:h_idx+h_slice]
            h_tile = h_prev[:, h_idx:h_idx+h_slice, p_idx:p_idx+p_slice, n_idx:n_idx+n_slice]
            # print_tensor_fp16_hex_inline(h_tile)
            # tile tensor로 변경해서 연산
            dx_tile = torch.einsum("bh, bhp -> bhp", dt_tile, x_tile)
            dxB_tile = torch.einsum("bhp, bn -> bhpn", dx_tile, B_tile)
            # dBx_tile = torch.einsum("bh, bn, bhp -> bhpn", dt_tile, B_tile, x_tile)
            # save_tensor_as_hex(dBx_tile, f"{base_path}/0_dBx_python.hex")
            # print('dx: ')
            # print_tensor_fp16_hex_inline(dx_tile)
            
            h_new = h_tile * rearrange(dA_tile, "b h -> b h 1 1") + dxB_tile
            # save_tensor_as_hex(h_new, f"{base_path}/0_h_new_python.hex")
            # print('h_new: ')
            # print_tensor_fp16_hex_inline(h_new)

            y = torch.einsum("bhpn, bn -> bhp", h_new, C_tile)
            # if h_idx == 0 and p_idx == 0:
                # save_tensor_as_hex(y, f"{base_path}/0_hc_python.hex")
            # print('y: ')
            # print_tensor_fp16_hex_inline(y)

            y = y + rearrange(D_tile, "h -> h 1") * x_tile
            # print_tensor_fp16_hex_inline(y)
            

            Y[:, h_idx:h_idx+h_slice, p_idx:p_idx+p_slice] += y

Y = rearrange(Y, "b h p -> b (h p)")

# 결과 저장
save_tensor_as_hex(Y, f"{base_path}/0_y_out_python_fp16.hex")
# print("(❁´◡`❁) 연산 완료! 결과 저장 위치:", f"{base_path}/0_y_out_python.hex")
# # print("y_Python =\n", Y.view(H_, P_))
# y_out = load_hex_tensor(f"{base_path}/0_y_out.hex", (B_, H_, P_))
# # print("\ny_Hardware =\n", y_out.view(H_, P_))



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

def compare_fp32_fp16_hex(fp32_file: str, fp16_file: str, topk: int = 10, atol: float = 1e-3):
    """
    FP32 .hex(8헥사/라인) vs FP16 .hex(4헥사/라인) 비교.
    - 비교는 공정하게 FP32 기준에서 수행: FP16을 FP32로 캐스팅하여 차이 계산.
    - 출력: 총 개수, Max/Mean/RMS 오차, 임계치 초과 개수, Top-K 샘플
    """
    t32 = load_fp32_hex(fp32_file)                     # torch.float32
    t16 = load_fp16_hex(fp16_file).to(torch.float32)   # torch.float32로 승격하여 비교

    if t32.numel() != t16.numel():
        raise ValueError(f"파일 길이가 다릅니다: {t32.shape} vs {t16.shape}")

    diff = (t32 - t16).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    rms_error = torch.sqrt(torch.mean((t32 - t16) ** 2)).item()
    num_errors = (diff > atol).sum().item()

    print(f"🔍 Total elements: {t32.numel()}")
    print(f"📊 Max abs error : {max_error}")
    print(f"📊 Mean abs error: {mean_error}")
    print(f"📐 RMS error: {rms_error}")
    print(f"⚠️  Elements with >{atol} error: {num_errors}")

    k = min(topk, t32.numel())
    if k > 0:
        top = torch.topk(diff, k=k)
        print("\nTop {} max error entries:".format(k))
        for idx, val in zip(top.indices, top.values):
            i = idx.item()
            print(f"[{i}] FP32={t32[i].item():.6f}, FP16={t16[i].item():.6f}, AbsErr={val.item():.6f}")

    # 유용하게 쓰라고 요약 dict도 반환
    return {
        "total": t32.numel(),
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "rms_error": rms_error,
        "num_exceed_atol": num_errors,
        "atol": atol,
        "topk": k,
    }

# 사용 예시
file_fp32 = f"{base_path}/0_y_fp32.hex"
file_fp16 =  f"{base_path}/0_y_out_python_fp16.hex"
compare_fp32_fp16_hex(file_fp32, file_fp16)