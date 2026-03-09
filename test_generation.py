import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint_path = "checkpoints/sleep_mixed_sft_llama3_8b/global_step_145_hf"

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, legacy=False)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

test_prompt = "You are a sleep medicine expert. The user's average sleep duration is 6 hours. Instruction: Provide insights."

messages = [{"role": "user", "content": test_prompt}]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

print("="*80)
print("Input text (after chat template):")
print("="*80)
print(input_text)
print("="*80)

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated text (full):")
print("="*80)
print(generated_text)
print("="*80)

print("\nExtracted response (current method):")
print("="*80)
response = generated_text[len(input_text):].strip()
print(response)
print("="*80)

print(f"\nInput length: {len(input_text)}")
print(f"Generated length: {len(generated_text)}")
print(f"Response length: {len(response)}")
