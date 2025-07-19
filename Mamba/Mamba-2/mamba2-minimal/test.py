import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba2 import Mamba2LMHeadModel, InferenceCache, Mamba2Config

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'state-spaces/mamba2-130m'  # "AntonV/mamba2-130m-hf"

# 모델/토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token_id = tokenizer.eos_token_id

config = Mamba2Config(d_model=768, n_layer=24, vocab_size=50277)
model = Mamba2LMHeadModel(config)
model.load_state_dict(torch.load(r"C:\Internship\mamba2-130m\mamba2_130m_quantized.pth"))
model.to(device)

# 입력 프롬프트
prompt = "The future of AI"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)  # (1, L)
prefix, tokens = input_ids[:, :-1], input_ids[:, -1:]

# 캐시 초기화
chunk_size = model.args.chunk_size
n_chunked = (prefix.shape[1] // chunk_size) * chunk_size
if n_chunked > 0:
    _, h = model(prefix[:, :n_chunked], None)
else:
    h = [InferenceCache.alloc(1, model.args, device=device) for _ in range(model.args.n_layer)]
for i in range(n_chunked, prefix.shape[1]):
    _, h = model(prefix[:, i:i+1], h)

# 생성
generated = [t.item() for t in input_ids[0]]
for _ in range(20):
    with torch.no_grad():
        out, h = model(tokens, h)
        # print(out[0,0,:5])
    logits = out[0, -1]
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    if next_token.item() == tokenizer.eos_token_id:
        break
    generated.append(next_token.item())
    tokens = next_token.unsqueeze(0)

# 출력
print(tokenizer.decode(generated, skip_special_tokens=True))
