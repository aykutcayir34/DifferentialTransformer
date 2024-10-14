import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len: int, *, device: torch.device):
        seq = torch.arange(max_seq_len, device=device)
        freqs = torch.einsum("i,j->ij", seq, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

class DifferentialAttention(nn.Module):
    def __init__(self, dim_model: int, head_nums: int, depth: int):
        super().__init__()
        
        self.head_dim = dim_model // head_nums

        self.Q = nn.Linear(dim_model, 2 * self.head_dim, bias=False)
        self.K = nn.Linear(dim_model, 2 * self.head_dim, bias=False)
        self.V = nn.Linear(dim_model, 2 * self.head_dim, bias=False)
        self.scale = self.head_dim ** -0.5
        self.depth = depth
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.rotary_emb = RotaryEmbedding(self.head_dim * 2)

    def forward(self, x):
        lambda_init = lambda_init_fn(self.depth)
        Q = self.Q(x)
        K = self.K(x)

        seq_len = x.shape[1]
        cos, sin = self.rotary_emb(seq_len, device=x.device)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
    
        Q1, Q2 = Q.chunk(2, dim=-1)
        K1, K2 = K.chunk(2, dim=-1)
        V = self.V(x)
        A1 = Q1 @ K1.transpose(-2, -1) * self.scale
        A2 = Q2 @ K2.transpose(-2, -1) * self.scale
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(Q1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(Q2)
        lambda_ = lambda_1 - lambda_2 + lambda_init
        return (F.softmax(A1, dim=-1)  - lambda_ * F.softmax(A2, dim=-1)) @ V

class MultiHeadDifferentialAttention(nn.Module):
    def __init__(self, dim_model: int, head_nums: int, depth: int):
        super().__init__()
        self.heads = nn.ModuleList([DifferentialAttention(dim_model, head_nums, depth) for _ in range(head_nums)])
        self.group_norm = RMSNorm(dim_model)
        self.output = nn.Linear(2 * dim_model, dim_model, bias=False)
        self.lambda_init = lambda_init_fn(depth)
    
    def forward(self, x):
        o = torch.cat([self.group_norm(h(x)) for h in self.heads], dim=-1)
        o = o * (1 - self.lambda_init)
        return self.output(o)
    
class DifferentialTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int = 8, head_dim: int = 64, vocab_size: int = 10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.layers = nn.ModuleList([
            MultiHeadDifferentialAttention(dim, heads, depth_idx)
            for depth_idx in range(depth)
        ])
        self.ln1 = RMSNorm(dim)
        self.ln2 = RMSNorm(dim)
        self.ffn = FeedForward(dim, (dim // 3) * 8)
        self.output = nn.Linear(dim, self.vocab_size)
    
    def forward(self, x):
        for attn in self.layers:
            y = attn(self.ln1(x)) + x
            x = self.ffn(self.ln2(y)) + y
        return self.output(x)

x = torch.randn(1, 10, 128)
model = DifferentialTransformer(128, 12)
out = model(x)
print(out.shape)
