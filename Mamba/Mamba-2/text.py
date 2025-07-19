from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("AntonV/mamba2-130m-hf")  # 2.7b
model = AutoModelForCausalLM.from_pretrained("AntonV/mamba2-130m-hf")

print('\n==================================================\n')
print(model)
print('\n==================================================\n')

# input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"]
# out = model.generate(input_ids, max_new_tokens=10)
# print(tokenizer.batch_decode(out))

# print('\n==================================================\n')