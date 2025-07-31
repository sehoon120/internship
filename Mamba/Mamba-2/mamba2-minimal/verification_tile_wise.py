import torch
import numpy as np
from einops import rearrange

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
H_ = 16
P_ = 4
N_ = 16

# B_ = 1
# H_ = 24
# P_ = 64
# N_ = 128

h_slice = 4
p_slice = 4
n_slice = 16  # 128  # 이 축으로는 slice 불가

# 경로 지정
base_path = "C:/Internship/intermediate_datas"
dA = load_hex_tensor(f"{base_path}/0_dA_copy.hex", (B_, H_))
dt = load_hex_tensor(f"{base_path}/0_dt_copy.hex", (B_, H_))
x = load_hex_tensor(f"{base_path}/0_x_copy.hex", (B_, H_, P_))
B = load_hex_tensor(f"{base_path}/0_B_copy.hex", (B_, N_))
C = load_hex_tensor(f"{base_path}/0_C_copy.hex", (B_, N_))
D = load_hex_tensor(f"{base_path}/0_D_copy.hex", (H_,))
h_prev = load_hex_tensor(f"{base_path}/0_ssm_state_copy.hex", (B_, H_, P_, N_))

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
            dBx_tile = torch.einsum("bh, bn, bhp -> bhpn", dt_tile, B_tile, x_tile)
            # save_tensor_as_hex(dBx_tile, f"{base_path}/0_dBx_python.hex")
            # print('dBx: ')
            # print_tensor_fp16_hex_inline(dBx_tile)
            
            h_new = h_tile * rearrange(dA_tile, "b h -> b h 1 1") + dBx_tile
            # save_tensor_as_hex(h_new, f"{base_path}/0_h_new_python.hex")
            # print('h_new: ')
            # print_tensor_fp16_hex_inline(h_new)

            y = torch.einsum("bhpn, bn -> bhp", h_new, C_tile)
            # save_tensor_as_hex(y, f"{base_path}/0_hc_python.hex")
            # print('y: ')
            # print_tensor_fp16_hex_inline(y)

            y = y + rearrange(D_tile, "h -> h 1") * x_tile
            # print_tensor_fp16_hex_inline(y)
            

            Y[:, h_idx:h_idx+h_slice, p_idx:p_idx+p_slice] += y

Y = rearrange(Y, "b h p -> b (h p)")

# 결과 저장
save_tensor_as_hex(Y, f"{base_path}/0_y_out_copy_python.hex")
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

# 사용 예시
file_py = f"{base_path}/0_y_out_copy_python.hex"
file_v =  f"{base_path}/0_y_out_copy.hex"
compare_fp16_hex(file_py, file_v)