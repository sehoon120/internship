import torch, gc
from transformers import AutoTokenizer
from datasets import load_dataset
from mamba2 import Mamba2LMHeadModel, Mamba2Config, ssd, InferenceCache
from tqdm import tqdm
import os
import torch.nn.functional as F
from PTQ_mamba import Mamba_Block


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.is_available())          # ✅ Truee 여야 함
#print(torch.cuda.device_count())          # ex. 1
#print(torch.cuda.get_device_name(0))      # 예: NVIDIA RTX 3090
#print(device)                             # 예: cuda, cuda:0
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


from FXP_simulator import FXP16Simulator, FXP32Simulator, FXP8Simulator
# 8-bit FXP 시뮬레이터
fxp8_2 = FXP8Simulator(frac_bits=2)
fxp8_3 = FXP8Simulator(frac_bits=3)
fxp8_4 = FXP8Simulator(frac_bits=4)
fxp8_5 = FXP8Simulator(frac_bits=5)
fxp8_6 = FXP8Simulator(frac_bits=6)
fxp8_7 = FXP8Simulator(frac_bits=7)
# 16-bit FXP 시뮬레이터
fxp16_4 = FXP16Simulator(frac_bits=4)
fxp16_5 = FXP16Simulator(frac_bits=5)
fxp16_6 = FXP16Simulator(frac_bits=6)
fxp16_7 = FXP16Simulator(frac_bits=7)
fxp16_8 = FXP16Simulator(frac_bits=8)
fxp16_9 = FXP16Simulator(frac_bits=9)
fxp16_10 = FXP16Simulator(frac_bits=10)
fxp16_11 = FXP16Simulator(frac_bits=11)
fxp16_12 = FXP16Simulator(frac_bits=12)
fxp16_13 = FXP16Simulator(frac_bits=13)
fxp16_14 = FXP16Simulator(frac_bits=14)
fxp16_15 = FXP16Simulator(frac_bits=15)
# 32-bit FXP 시뮬레이터
fxp32_16 = FXP32Simulator(frac_bits=16)
fxp32_18 = FXP32Simulator(frac_bits=18)
fxp32_20 = FXP32Simulator(frac_bits=20)
fxp32_21 = FXP32Simulator(frac_bits=21)
fxp32_24 = FXP32Simulator(frac_bits=24)

def q_dq(x, a, b):
    if a == 8:
        if b == 2:
            return fxp8_2.dequantize(fxp8_2.quantize(x))
        elif b == 3:
            return fxp8_3.dequantize(fxp8_3.quantize(x))
        elif b == 4:
            return fxp8_4.dequantize(fxp8_4.quantize(x))
        elif b == 5:
            return fxp8_5.dequantize(fxp8_5.quantize(x))
        elif b == 6:
            return fxp8_6.dequantize(fxp8_6.quantize(x))
        elif b == 7:
            return fxp8_7.dequantize(fxp8_7.quantize(x))

    elif a == 16:
        if b == 4:
            return fxp16_4.dequantize(fxp16_4.quantize(x))
        elif b == 5:
            return fxp16_5.dequantize(fxp16_5.quantize(x))
        elif b == 6:
            return fxp16_6.dequantize(fxp16_6.quantize(x))
        elif b == 7:
            return fxp16_7.dequantize(fxp16_7.quantize(x))
        elif b == 8:
            return fxp16_8.dequantize(fxp16_8.quantize(x))
        elif b == 9:
            return fxp16_9.dequantize(fxp16_9.quantize(x))
        elif b == 10:
            return fxp16_10.dequantize(fxp16_10.quantize(x))
        elif b == 11:
            return fxp16_11.dequantize(fxp16_11.quantize(x))
        elif b == 12:
            return fxp16_12.dequantize(fxp16_12.quantize(x))
        elif b == 13:
            return fxp16_13.dequantize(fxp16_13.quantize(x))
        elif b == 14:
            return fxp16_14.dequantize(fxp16_14.quantize(x))
        elif b == 15:
            return fxp16_15.dequantize(fxp16_15.quantize(x))

    elif a == 32:
        if b == 16:
            return fxp32_16.dequantize(fxp32_16.quantize(x))
        elif b == 18:
            return fxp32_18.dequantize(fxp32_18.quantize(x))
        elif b == 20:
            return fxp32_20.dequantize(fxp32_20.quantize(x))
        elif b == 21:
            return fxp32_21.dequantize(fxp32_21.quantize(x))
        elif b == 24:
            return fxp32_24.dequantize(fxp32_24.quantize(x))

    raise ValueError(f"Unsupported FXP format: {a} total bits with {b} fractional bits")

current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, '../../..')
model_path = os.path.join(path, 'mamba2-1.3b/mamba2_1.3b_quantized.pth')


print(torch.cuda.memory_summary())


# ⬇️ 너의 모델 로드 경로 수정
#config = Mamba2Config(d_model=2560, n_layer=64, vocab_size=50288)
config = Mamba2Config(d_model=2048, n_layer=48, vocab_size=50277)
model = Mamba2LMHeadModel(config)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id

# ⬇️ WinoGrande dataset (소형 버전 또는 dev set)
dataset = load_dataset("winogrande", "winogrande_xs", split="validation")

correct = 0
total = 0

print("Evaluating quantized Mamba on WinoGrande...")

for sample in tqdm(dataset):
    sent = sample["sentence"]
    option1 = sample["option1"]
    option2 = sample["option2"]
    answer = sample["answer"]  # "1" or "2"

    prompt = f"{sent}\nQ: What does 'it' refer to?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    print(torch.cuda.memory_summary())

# ==================== Generate Start ====================
    h = [InferenceCache.alloc(
        batch_size=1,
        args=config,
        device=device
    ) for _ in range(config.n_layer)]

    prefix, f_token = input_ids[:, :-1], input_ids[:, -1:]
    chunk_size = model.args.chunk_size
    n_chunked = (prefix.shape[1] // chunk_size) * chunk_size
    if n_chunked > 0:
        _, h = model(prefix[:, :n_chunked], None)
    else:
        h = [InferenceCache.alloc(1, model.args, device=device) for _ in range(model.args.n_layer)]
    for i in range(n_chunked, prefix.shape[1]):
        _, h = model(prefix[:, i:i+1], h)

    generated = [t.item() for t in input_ids[0]]
    
    with torch.no_grad():
        for t in range(5):  # 5 단어 생성
            seqlen = f_token.shape[1]
            input_tensor = f_token.to(device)
            u = model.backbone['embedding'](input_tensor)
            residual = u  
            residual = q_dq(residual, 32, 16)
            for i in range(config.n_layer):
                residual, h = Mamba_Block(model=model, residual=residual, h=h, config=config, i=i)

            residual = model.backbone.norm_f(residual)
            logits = model.lm_head(residual)  # shape: (1, 1, vocab_size)
            out = logits[:, :seqlen]  # seqlen=1
            logits = out[0, -1]  # 최종 토큰의 로짓
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # 종료 조건 확인 (EOS 토큰일 경우)
            if next_token.item() == tokenizer.eos_token_id:
                break
            # 다음 루프 준비
            generated.append(next_token.item())
            # f_token = next_token.unsqueeze(0)
            f_token = next_token.to(device).unsqueeze(0)
        
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
# ==================== Generate End ====================

    # 정답 포함 여부 판단
    pred_label = None
    if option1.lower() in decoded:
        pred_label = "1"
    elif option2.lower() in decoded:
        pred_label = "2"

    if pred_label == answer:
        correct += 1
    total += 1
    del input_ids, output, decoded
    torch.cuda.empty_cache()
    gc.collect()

accuracy = correct / total * 100
print(f"✅ WinoGrande Accuracy: {accuracy:.2f}% ({correct}/{total})")

