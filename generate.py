from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load your fine-tuned model
print("Loading your fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained("./gosense-gpt-final")
model = AutoModelForCausalLM.from_pretrained("./gosense-gpt-final")
model = model.to(device)
model.eval()
print("Model loaded.")

def generate(prompt, max_new_tokens=100, temperature=0.8):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test with your voice
prompts = [
    "Fraud rings are built",
    "The problem with UPI fraud is",
    "Every time I looked at the data",
]

for prompt in prompts:
    print(f"\n--- PROMPT: {prompt} ---\n")
    print(generate(prompt))
    print()