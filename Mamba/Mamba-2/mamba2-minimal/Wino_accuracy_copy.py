"""
    WinoGrande dataset으로 모델 정확도 평가하기
    19개마다 메모리가 터짐. 연속 실행으로 이어갈 수 있게 하였으나, 장기적으로는 수정이 필요함.

    문장 option 2개로 loss-prob 비교하기
    generate가 아닌 forward를 하도록 바꾸기
"""
import os, gc, torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from mamba2 import Mamba2LMHeadModel, Mamba2Config, ssd, InferenceCache
from PTQ_mamba import Mamba_Block
from FXP_simulator import FXP8Simulator, FXP16Simulator, FXP32Simulator
# import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

# FXP 시뮬레이터 정의 (생략 없이 전체)
fxp8_6 = FXP8Simulator(frac_bits=6)
fxp16_11 = FXP16Simulator(frac_bits=11)
fxp32_16 = FXP32Simulator(frac_bits=16)

def q_dq(x, a, b):
    if a == 8 and b == 6:
        return fxp8_6.dequantize(fxp8_6.quantize(x))
    elif a == 16 and b == 11:
        return fxp16_11.dequantize(fxp16_11.quantize(x))
    elif a == 32 and b == 16:
        return fxp32_16.dequantize(fxp32_16.quantize(x))
    raise ValueError(f"Unsupported FXP format: {a} total bits with {b} fractional bits")

# 🔧 경로 처리 (Windows & Linux 모두 호환)
current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.normpath(os.path.join(current_dir, '..', '..', '..', '..'))
model_path = os.path.join(path, 'mamba2-2.7b', 'mamba2_2.7b_quantized_FP.pth')
model_path = os.path.normpath(model_path)

log_path = r'C:\Internship\internship\Mamba\Mamba-2\mamba2-minimal\log\Wino_log.txt'
start_index, correct, total = 0, 0, 0
try:
    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in reversed(lines):
            if "Accuracy:" in line:
                parts = line.strip().split()
                # 예: [19] Accuracy: 57.89% (11/19)
                start_index = int(parts[0][1:-1])  # [19] → 19
                correct_total = parts[-1][1:-1]    # (11/19) → '11/19'
                correct, total = map(int, correct_total.split("/"))
                break
except FileNotFoundError:
    print("📁 로그 파일 없음: 처음부터 시작합니다.")
f = open(log_path, 'a')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"[ERROR] Model file does not exist: {model_path}")

# 모델 구성 및 로드
config = Mamba2Config(d_model=2560, n_layer=64, vocab_size=50288)
model = Mamba2LMHeadModel(config)
ckpt = torch.load(model_path, map_location=device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model = model.to(device).eval()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id

dataset = load_dataset("winogrande", "winogrande_xs", split="validation")
print("Evaluating quantized Mamba on WinoGrande...")
print(dataset[0])

a = 0
try:
    for sample in tqdm(dataset):
        if a < start_index:
            a += 1
            continue
        sent, option1, option2, answer = sample["sentence"], sample["option1"], sample["option2"], sample["answer"]
        prompt = f"{sent}\nQ: What does 'it' refer to?\nA:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Inference cache 초기화
        h = [InferenceCache.alloc(batch_size=1, args=config, device=device) for _ in range(config.n_layer)]

        prefix, f_token = input_ids[:, :-1], input_ids[:, -1:]
        chunk_size = model.args.chunk_size
        n_chunked = (prefix.shape[1] // chunk_size) * chunk_size
        if n_chunked > 0:
            _, h = model(prefix[:, :n_chunked], None)
        for i in range(n_chunked, prefix.shape[1]):
            _, h = model(prefix[:, i:i+1], h)

        generated = [t.item() for t in input_ids[0]]

        with torch.no_grad():
            for t in range(5):
                seqlen = f_token.shape[1]
                input_tensor = f_token.to(device)
                u = model.backbone['embedding'](input_tensor)
                residual = q_dq(u, 32, 16)

                for i in range(config.n_layer):
                    residual, h = Mamba_Block(model=model, residual=residual, h=h, config=config, i=i)

                residual = model.backbone.norm_f(residual)
                logits = model.lm_head(residual)
                probs = F.softmax(logits[0, -1], dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

                generated.append(next_token.item())
                f_token = next_token.to(device).unsqueeze(0)

        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        decoded = decoded.lower()
        pred_label = "1" if option1.lower() in decoded else ("2" if option2.lower() in decoded else None)

        if pred_label == answer:
            correct += 1
        total += 1
        accuracy = correct / total * 100
        a += 1
        f.write(f"[{a}] Accuracy: {accuracy:.2f}% ({correct}/{total})\n")
        del input_ids, residual, h, logits, probs, f_token, next_token , decoded
        torch.cuda.empty_cache()
        gc.collect()
except Exception as e:
    # f.write(f"[ERROR at sample {a}]: {e}\n")
    del model, dataset, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    f.close()
    raise e

del model, dataset, tokenizer
torch.cuda.empty_cache()
gc.collect()
f.close()
accuracy = correct / total * 100
print(f"\n\n✅ WinoGrande Accuracy: {accuracy:.2f}% ({correct}/{total})")
