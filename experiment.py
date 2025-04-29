import torch
import torch.nn as nn
from model import Block
import argparse

torch.manual_seed(69)

if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--kv", type=int, default=0)
    args = parser.parse_args()

    model = Block(args.d_model, args.n_heads)
    x = torch.randn(2, 5, args.d_model)

    if args.kv == 0:
        out_full, k_full, v_full, attn_matrix = model(x)
        print("Output (no cache):", out_full.shape)
        print("Attention matrix shape:", attn_matrix.shape)
        print("Attention matrix:\n", attn_matrix)
    elif args.kv == 1:
        past_k, past_v = None, None
        outputs = []
        for t in range(x.size(1)):
            x_step = x[:, t:t + 1, :]
            out_step, past_k, past_v, attn_matrix = model(x_step, past_k, past_v)
            outputs.append(out_step)
            print(f"Step {t} attention matrix shape:", attn_matrix.shape)
            print(f"Step {t} attention matrix:\n", attn_matrix)
        out_cached = torch.cat(outputs, dim=1)
        print("Output (with cache):", out_cached.shape)