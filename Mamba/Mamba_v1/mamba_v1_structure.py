# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch, os

# model_id = "state-spaces/mamba-130m-hf"
# tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")

# out = model.generate(**tok("hello", return_tensors="pt"), max_new_tokens=32)
# print(tok.decode(out[0], skip_special_tokens=True))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os
import torch.nn as nn
import types
from collections import defaultdict

# ============================ CONFIG ============================
MODEL_ID = "state-spaces/mamba-130m-hf"
NUM_THREADS = 4
MAX_NEW_TOKENS = 8           # decode 스텝 길이
PRINT_LIMIT_PER_PHASE = 80   # 각 phase별로 요약 출력 개수
CAPTURE_VALUES = False       # True면 일부 값 샘플도 저장(느려질 수 있음)
MAX_VALUE_SAMPLES = 16
SAVE_JSON = True             # 캡처 결과를 json으로 저장
JSON_PATH = "mamba_v1_cpu_forward_captures.json"
# mixer 내부만 특정 블록으로 좁히고 싶다면 아래에 일부 이름 접두사 지정 (예: "backbone.layers.4.mixer")
FOCUS_PREFIX = ""            # 빈 문자열이면 전체에 훅

# ============================ SETUP =============================
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os, json
import torch.nn as nn

os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
torch.set_num_threads(NUM_THREADS)

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to("cpu").eval()

# ======================== HOOK UTILITIES ========================
captures = []               # 모든 캡처(phase 구분 포함)
CURRENT_PHASE = None        # "prefill" / "decode" 라벨

def _shape_of(x):
    return tuple(x.shape) if isinstance(x, torch.Tensor) else str(type(x))

def _sample_vals(x):
    if not CAPTURE_VALUES or not isinstance(x, torch.Tensor):
        return None
    with torch.no_grad():
        flat = x.detach().cpu().flatten()
        return flat[:MAX_VALUE_SAMPLES].tolist()

def make_hook(full_name):
    def hook(module, inputs, output):
        ins = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        outs = output if isinstance(output, (list, tuple)) else [output]
        rec = {
            "phase": CURRENT_PHASE,
            "name": full_name,
            "type": module.__class__.__name__,
            "in_shapes": [_shape_of(t) for t in ins],
            "out_shapes": [_shape_of(t) for t in outs],
        }
        if CAPTURE_VALUES:
            rec["in_sample"]  = [_sample_vals(t) for t in ins]
            rec["out_sample"] = [_sample_vals(t) for t in outs]
        captures.append(rec)
    return hook

def name_wanted(name: str) -> bool:
    """mixer 내부(in_proj/conv/scan/out_proj/norm 등)까지 포함하도록 넓게 필터"""
    nl = name.lower()
    if FOCUS_PREFIX and not nl.startswith(FOCUS_PREFIX.lower()):
        return False
    keywords = [
        "mamba", "mixer",
        "in_proj", "out_proj",
        "conv", "conv1d",
        "norm", "rmsnorm", "layernorm",
        "scan", "selective", "ssd",
        "backbone", "layers",
        "embedding", "lm_head",
    ]
    if any(k in nl for k in keywords):
        return True
    return False

TARGET_TYPES = (nn.Embedding, nn.Linear, nn.Conv1d, nn.LayerNorm)
# RMSNorm 클래스가 커스텀일 수 있어 이름으로 포착: name_wanted()에서 잡아냄

def register_hooks(m):
    hooks = []
    for name, module in m.named_modules():
        try:
            if isinstance(module, TARGET_TYPES) or name_wanted(name):
                hooks.append(module.register_forward_hook(make_hook(name)))
        except Exception:
            pass
    return hooks

def print_phase_summary(phase_name: str, limit=50):
    rows = [r for r in captures if r["phase"] == phase_name]
    print(f"\n=== {phase_name.upper()} (count={len(rows)}) ===")
    for i, rec in enumerate(rows[:limit]):
        print(f"{i:03d} {rec['name']} ({rec['type']})")
        print("    in :", rec['in_shapes'])
        print("    out:", rec['out_shapes'])

# ======================== PREFILL CAPTURE =======================
hooks = register_hooks(model)
CURRENT_PHASE = "prefill"

prompt = """
Mamba is a selective state-space model. In simple terms, a
Mamba is a selective state-space model. In simple terms, a
Mamba is a selective state-space model. In simple terms, a
Mamba is a selective state-space model. In simple terms, a
Mamba is a selective state-space model. In simple terms, a
Mamba is a selective state-space model. In simple terms, a
"""
inp = tok(prompt, return_tensors="pt")  # CPU
with torch.no_grad():
    out = model(**inp, use_cache=True, return_dict=True)

for h in hooks: h.remove()

# ========================= DECODE CAPTURE =======================
hooks = register_hooks(model)
CURRENT_PHASE = "decode"

past = out.past_key_values if hasattr(out, "past_key_values") else out.get("past_key_values", None)
# 간단한 greedy 디코딩 루프
with torch.no_grad():
    next_id = out.logits[:, -1:].argmax(dim=-1)
    gen_tokens = [next_id]
    for _ in range(MAX_NEW_TOKENS - 1):
        out = model(input_ids=next_id, past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values if hasattr(out, "past_key_values") else out.get("past_key_values", None)
        next_id = out.logits[:, -1:].argmax(dim=-1)
        gen_tokens.append(next_id)

for h in hooks: h.remove()

# ======================= PRINT / SAVE RESULT ====================
print_phase_summary("prefill", PRINT_LIMIT_PER_PHASE)
print_phase_summary("decode", PRINT_LIMIT_PER_PHASE)

if SAVE_JSON:
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(captures, f, indent=2)
    print(f"\nSaved captures to {JSON_PATH}")

# (선택) 디코드 결과 문장 보기
try:
    full = torch.cat([inp["input_ids"], torch.stack(gen_tokens, dim=1)], dim=1)
    print("\nGenerated:\n", tok.decode(full[0], skip_special_tokens=True))
except Exception:
    pass


'''
=== PREFILL (count=244) ===
001 backbone.layers.0.norm (MambaRMSNorm)
    in : [(1, 15, 768)]
    out: [(1, 15, 768)]
002 backbone.layers.0.mixer.in_proj (Linear)
    in : [(1, 15, 768)]
    out: [(1, 15, 3072)]
003 backbone.layers.0.mixer.conv1d (Conv1d)
    in : [(1, 1536, 15)]
    out: [(1, 1536, 18)]
004 backbone.layers.0.mixer.act (SiLU)
    in : [(1, 1536, 15)]
    out: [(1, 1536, 15)]
005 backbone.layers.0.mixer.x_proj (Linear)
    in : [(1, 15, 1536)]
    out: [(1, 15, 80)]
006 backbone.layers.0.mixer.dt_proj (Linear)
    in : [(1, 15, 48)]
    out: [(1, 15, 1536)]
007 backbone.layers.0.mixer.act (SiLU)
    in : [(1, 1536, 15)]
    out: [(1, 1536, 15)]
008 backbone.layers.0.mixer.out_proj (Linear)
    in : [(1, 15, 1536)]
    out: [(1, 15, 768)]
009 backbone.layers.0.mixer (MambaMixer)
    in : [(1, 15, 768)]
    out: [(1, 15, 768)]
010 backbone.layers.0 (MambaBlock)
    in : [(1, 15, 768)]
    out: [(1, 15, 768)]


=== DECODE (count=1708) ===
001 backbone.layers.0.norm (MambaRMSNorm)
    in : [(1, 1, 768)]
    out: [(1, 1, 768)]
002 backbone.layers.0.mixer.in_proj (Linear)
    in : [(1, 1, 768)]
    out: [(1, 1, 3072)]
003 backbone.layers.0.mixer.conv1d (Conv1d)
    in : [(1, 1536, 1)]
    out: [(1, 1536, 4)]
004 backbone.layers.0.mixer.act (SiLU)
    in : [(1, 1536, 1)]
    out: [(1, 1536, 1)]
005 backbone.layers.0.mixer.x_proj (Linear)
    in : [(1, 1, 1536)]
    out: [(1, 1, 80)]
006 backbone.layers.0.mixer.dt_proj (Linear)
    in : [(1, 1, 48)]
    out: [(1, 1, 1536)]
007 backbone.layers.0.mixer.act (SiLU)
    in : [(1, 1536, 1)]
    out: [(1, 1536, 1)]
008 backbone.layers.0.mixer.out_proj (Linear)
    in : [(1, 1, 1536)]
    out: [(1, 1, 768)]
009 backbone.layers.0.mixer (MambaMixer)
    in : [(1, 1, 768)]
    out: [(1, 1, 768)]
010 backbone.layers.0 (MambaBlock)
    in : [(1, 1, 768)]
    out: [(1, 1, 768)]
'''