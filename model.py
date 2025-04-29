import torch
import torch.nn as nn
import torch.nn.functional as F



class Block(nn.Module):
    def __init__ (self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.h_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model needs to be div by nheads"

        self.q = nn.Linear(d_model, d_model)
        self.k  = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, k_ = None, v_ = None):
        B, T, C = x.size()
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.view(B, T, self.n_heads, self.h_dim).transpose(1,2) # should be B, nhead, T, hdim. transpose is to swap dim 1 and 2 of tensor
        k = k.view(B, T, self.n_heads, self.h_dim).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.h_dim).transpose(1,2)

        if k_ is not None and v_ is not None:
            k = torch.cat([k_, k], dim = 2) # cat along seq dim
            v = torch.cat([v_, v], dim = 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.h_dim ** 0.5) # should be   (B, n_heads, T, T_)
        attn_probs = F.softmax(attn_scores, dim = -1)
        attn_out = torch.matmul(attn_probs, v)  # should be  # (B, n_heads, T, h_dim)
        attn_output = attn_out.transpose(1, 2).contiguous().view(B, T, C) # concat heads
        out = self.out(attn_output)
        return out, k, v, attn_probs