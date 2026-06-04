import torch
import torch.nn as nn
from gpt import GPT

# Load your writing
with open("my_writing.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Total characters: {len(text):,}")

# Build character vocabulary
chars = sorted(set(text))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} unique characters")

# Character to integer and back
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode entire dataset
data = torch.tensor(encode(text), dtype=torch.long)

# Train / validation split
n = int(0.9 * len(data))
train = data[:n]
val = data[n:]

# Settings
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

seq_len = 64
batch_size = 16
d_model = 128
n_heads = 4
n_layers = 4

# Build model
model = GPT(vocab_size, d_model, n_heads, n_layers, seq_len).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Batch function
def get_batch(split):
    d = train if split == 'train' else val
    ix = torch.randint(len(d) - seq_len, (batch_size,))
    x = torch.stack([d[i:i+seq_len] for i in ix])
    y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# Loss estimation
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        split_losses = []
        for _ in range(50):
            x, y = get_batch(split)
            logits = model(x)
            B, T, V = logits.shape
            loss = nn.functional.cross_entropy(
                logits.view(B*T, V), y.view(B*T)
            )
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses

# Training loop
optimiser = torch.optim.Adam(model.parameters(), lr=3e-4)
n_steps = 5000

print("Starting training...")
for step in range(n_steps):
    if step % 500 == 0:
        losses = estimate_loss()
        print(f"Step {step}: train={losses['train']:.4f} val={losses['val']:.4f}")

    x, y = get_batch('train')
    logits = model(x)
    B, T, V = logits.shape
    loss = nn.functional.cross_entropy(
        logits.view(B*T, V), y.view(B*T)
    )
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimiser.step()

print("Training complete.")

# Generate some text
@torch.no_grad()
def generate(start_text, max_new_tokens=200, temperature=0.8):
    model.eval()
    context = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        ctx = context[:, -seq_len:]
        logits = model(ctx)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_id], dim=1)
    return decode(context[0].tolist())

print("\n--- GENERATED TEXT ---\n")
print(generate("The fraud ring", max_new_tokens=300))