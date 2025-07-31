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

B_ = 1
H_ = 16
P_ = 4
N_ = 16

# 경로 지정
base_path = "C:/Internship/intermediate_datas"
dA = load_hex_tensor(f"{base_path}/0_dA_copy.hex", (B_, H_))
dt = load_hex_tensor(f"{base_path}/0_dt_copy.hex", (B_, H_))
x = load_hex_tensor(f"{base_path}/0_x_copy.hex", (B_, H_, P_))
B = load_hex_tensor(f"{base_path}/0_B_copy.hex", (B_, N_))
C = load_hex_tensor(f"{base_path}/0_C_copy.hex", (B_, N_))
D = load_hex_tensor(f"{base_path}/0_D_copy.hex", (H_,))
h_prev = load_hex_tensor(f"{base_path}/0_ssm_state_copy.hex", (B_, H_, P_, N_))

# 연산
dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
save_tensor_as_hex(dBx, f"{base_path}/0_dBx_copy_python.hex")

h_new = h_prev * rearrange(dA, "b h -> b h 1 1") + dBx
save_tensor_as_hex(h_new, f"{base_path}/0_h_new_copy_python.hex")

y = torch.einsum("bhpn, bn -> bhp", h_new, C)
save_tensor_as_hex(y, f"{base_path}/0_hc_copy_python.hex")

y = y + rearrange(D, "h -> h 1") * x
y = rearrange(y, "b h p -> b (h p)")

# 결과 저장
save_tensor_as_hex(y, f"{base_path}/0_y_out_copy_python.hex")
print("(❁´◡`❁) 연산 완료! 결과 저장 위치:", f"{base_path}/0_y_out_copy_python.hex")
print("y_Python =\n", y.view(H_, P_))
y_out_copy = load_hex_tensor(f"{base_path}/0_y_out_copy.hex", (B_, H_, P_))
print("\ny_Hardware =\n", y_out_copy.view(H_, P_))


print()