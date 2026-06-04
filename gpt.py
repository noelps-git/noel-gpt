import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
    
    def forward(self, token_ids):
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        Q, K, V = qkv.split(C, dim=2)
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        out = weights @ V
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, seq_len):
        super().__init__()
        self.emb = EmbeddingLayer(vocab_size, d_model, seq_len)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, seq_len)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        x = self.emb(token_ids)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.head(x)
        return logits

# Test
vocab_size = 50257
d_model = 768
n_heads = 12
n_layers = 12
seq_len = 1024

model = GPT(vocab_size, d_model, n_heads, n_layers, seq_len)

token_ids = torch.tensor([[1, 2, 3, 4, 5],
                           [6, 7, 8, 9, 10]])

logits = model(token_ids)
print("Output shape:", logits.shape)
print("Total parameters:", sum(p.numel() for p in model.parameters()))