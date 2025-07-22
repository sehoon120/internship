import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba2 import Mamba2LMHeadModel, Mamba2Config
from transformers import AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 1. Weight quantization 함수
# -------------------------------
def quantize_weight(tensor, num_bits=8):
    qmin, qmax = -128, 127
    min_val, max_val = tensor.min().item(), tensor.max().item()  # ⬅️ .item() 추가
    scale = (max_val - min_val) / (qmax - qmin + 1e-8)
    zero_point = int(round(-min_val / scale))
    q_tensor = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax).to(torch.int8)
    return q_tensor, scale, zero_point

# -------------------------------
# 2. Activation quantization 함수
# -------------------------------
def quantize_activation(tensor, num_bits=8):
    qmin, qmax = -128, 127
    max_val = tensor.abs().max()
    scale = max_val / ((2 ** (num_bits - 1)) - 1 + 1e-8)
    q_tensor = torch.clamp(
        torch.round(tensor / scale), qmin, qmax
    ).to(torch.int8)
    return q_tensor, scale

# -------------------------------
# 3. LUT 생성 (예: SiLU, exp, 등)
# -------------------------------
def build_lut(func, input_range=(-6, 6), resolution=256):
    x = torch.linspace(*input_range, steps=resolution)
    y = func(x)
    return x, y

# -------------------------------
# 4. 양자화된 Linear 레이어
# -------------------------------
class QuantLinear(nn.Module):
    def __init__(self, float_linear):
        super().__init__()
        self.weight = nn.Parameter(torch.quantize_per_tensor(
            float_linear.weight.data, scale=0.1, zero_point=0, dtype=torch.qint8
        ).dequantize())  # 임시 양자화 예시
        self.bias = float_linear.bias
        self.out_features = float_linear.out_features
        self.in_features = float_linear.in_features

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)

# -------------------------------
# 5. 기본 모델 정의
# -------------------------------
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# -------------------------------
# 6. 양자화 모델 정의
# -------------------------------


class QuantConv1d(nn.Module):
    def __init__(self, float_conv: nn.Conv1d):
        super().__init__()
        q_weight, scale, zp = quantize_weight(float_conv.weight.data)
        self.register_buffer("q_weight", q_weight)
        self.scale = scale
        self.zp = zp
        self.bias = float_conv.bias
        self.stride = float_conv.stride
        self.padding = float_conv.padding
        self.dilation = float_conv.dilation
        self.groups = float_conv.groups

    def forward(self, x):
        x_q, x_scale = quantize_activation(x)
        w = self.q_weight.float()
        out = F.conv1d(x_q.float(), w, bias=self.bias, stride=self.stride,
                       padding=self.padding, dilation=self.dilation, groups=self.groups)
        return out * (self.scale * x_scale)


class QuantRMSNorm(nn.Module):
    def __init__(self, float_norm: nn.Module, lut_bits=8):
        super().__init__()
        self.eps = float_norm.eps
        self.weight = float_norm.weight
        self.lut_x, self.lut_y = self.build_rsqrt_lut(bits=lut_bits)

    def build_rsqrt_lut(self, bits=8):
        x_vals = torch.linspace(1e-5, 10.0, steps=2**bits)  # RMS range
        y_vals = 1.0 / torch.sqrt(x_vals)                   # rsqrt LUT
        return x_vals, y_vals

    def forward(self, x):
        rms = torch.mean(x**2, dim=-1, keepdim=True) + self.eps
        idx = torch.bucketize(rms, self.lut_x)
        idx = torch.clamp(idx, 0, len(self.lut_y) - 1)
        rsqrt_approx = self.lut_y[idx].squeeze(-1)
        normed = x * rsqrt_approx
        return normed * self.weight.unsqueeze(0)



# --- Mamba2 래퍼 ---
class QuantMamba2LMHeadModel(nn.Module):
    def __init__(self, float_model):
        super().__init__()
        self.args = float_model.args
        self.device = float_model.device

        # Quantized embedding (float 그대로 사용 가능)
        self.embedding = float_model.backbone["embedding"]

        # Quantized backbone.layers
        self.layers = nn.ModuleList()
        for layer in float_model.backbone["layers"]:
            mixer = layer["mixer"]
            quant_mixer = self.quantize_mixer(mixer)
            quant_norm = QuantRMSNorm(layer["norm"])
            self.layers.append(nn.ModuleDict({
                "mixer": quant_mixer,
                "norm": quant_norm,
            }))

        self.norm_f = QuantRMSNorm(float_model.backbone["norm_f"])
        self.lm_head = QuantLinear(float_model.lm_head)

    def quantize_mixer(self, mixer):
        mixer.in_proj = QuantLinear(mixer.in_proj)
        mixer.conv1d = QuantConv1d(mixer.conv1d)
        mixer.out_proj = QuantLinear(mixer.out_proj)
        return mixer

    def forward(self, input_ids, h=None):
        if h is None:
            h = [None for _ in range(self.args.n_layer)]

        x = self.embedding(input_ids)
        for i, layer in enumerate(self.layers):
            y, h[i] = layer["mixer"](layer["norm"](x), h[i])
            x = y + x
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, h


from collections import defaultdict

class ActivationStatsCollector:
    def __init__(self):
        self.activation_mins = defaultdict(lambda: float("inf"))
        self.activation_maxs = defaultdict(lambda: float("-inf"))
        self.handles = []

    def _hook(self, name):
        def fn(module, input, output):
            self.activation_mins[name] = min(self.activation_mins[name], output.min().item())
            self.activation_maxs[name] = max(self.activation_maxs[name], output.max().item())
        return fn

    def register_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.ReLU, nn.SiLU)):
                handle = module.register_forward_hook(self._hook(name))
                self.handles.append(handle)

    def clear_hooks(self):
        for h in self.handles:
            h.remove()

    def get_stats(self):
        return {
            name: {
                "min": self.activation_mins[name],
                "max": self.activation_maxs[name],
                "scale": (self.activation_maxs[name] - self.activation_mins[name]) / 255.0,
                "zero_point": int(round(-self.activation_mins[name] / ((self.activation_maxs[name] - self.activation_mins[name]) / 255.0 + 1e-8)))
            }
            for name in self.activation_mins
        }


# -------------------------------
# 7. 테스트
# -------------------------------
# float32 모델 생성
# config = Mamba2Config(d_model=768, n_layer=24, vocab_size=50277)
model_name = 'state-spaces/mamba2-130m'  # "AntonV/mamba2-130m-hf"
model_fp32 = Mamba2LMHeadModel.from_pretrained(model_name, device=device)
model_fp32.eval()
collector = ActivationStatsCollector()
collector.register_hooks(model_fp32)

# 양자화 모델 생성
model_quant = QuantMamba2LMHeadModel(model_fp32).eval()

# test
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id
# prompt = """
# Mamba is a new sequence model that can replace transformers in some cases. 
# It uses state space models instead of attention. Its advantage is that it is faster and more memory-efficient.
# Write a clear summary of how Mamba differs from Transformers.

# """
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)  # (1, L)
# output = model_quant(input_ids)

with torch.no_grad():
    for _ in range(10):
        inputs = tokenizer("Mamba replaces attention with state space models.", return_tensors="pt").to(model_fp32.device)
        _ = model_fp32(**inputs)

# 3. Hook 제거 및 결과 출력
collector.clear_hooks()
stats = collector.get_stats()

for layer, val in stats.items():
    print(f"{layer}: scale={val['scale']:.6f}, zp={val['zero_point']}, min={val['min']:.4f}, max={val['max']:.4f}")

# '''
# 입력 텐서
x = torch.randn(1, 16)

# 출력 비교
out_fp32 = model_fp32(x)
out_quant = model_quant(x)

print("Float32 Output:", out_fp32)
print("Quantized Output:", out_quant)
# '''