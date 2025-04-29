import torch
import torch.nn as nn
from model import Block
import argparse
import time
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
torch.manual_seed(69)

def measure_inference_time(model: nn.Module, seq_lens: list[int], d_model: int, use_kv_cache: bool = False):
    times = []
    for seq_len in seq_lens:
        x: torch.tensor = torch.randn(1, seq_len, d_model)
        k_ = v_ = None
        start = time.time()
        if use_kv_cache:
            for t in trange(seq_len):
                x_step = x[:, t:t+1, :]
                _, k_, v_, _ = model(x_step, k_, v_)
        else:
            for t in range(seq_len):
                _ = model(x[:, :t+1, :])
        elapsed = time.time() - start
        times.append(elapsed / seq_len)
    return times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--kv", type=int, default=0, help = "0 for no cache, 1 for yes cache, 2 for experiment")
    parser.add_argument('--max_seq_len', type=int, default = 64)
    args = parser.parse_args()

    model = Block(args.d_model, args.n_heads)
    x = torch.randn(1, 100, args.d_model)

    if args.kv == 0:
        out_full, k_full, v_full, attn_matrix = model(x)
        print("Output (no cache):", out_full.shape)
        print("Attention matrix shape:", attn_matrix.shape)
        print("Attention matrix:\n", attn_matrix)
    elif args.kv == 1:

        v_, k_ = None, None
        outputs = []
        for t in range(x.size(1)):
            x_step = x[:, t:t + 1, :]
            out_step, k_, v_, attn_matrix = model(x_step, k_, v_)
            outputs.append(out_step)
            print(f"Step {t} attention matrix shape:", attn_matrix.shape)
            print(f"Step {t} attention matrix:\n", attn_matrix)
        out_cached = torch.cat(outputs, dim=1)
        print("Output (with cache):", out_cached.shape)
    elif args.kv == 2:
        seq_lens = list(range(16, args.max_seq_len + 1, 16))
        model.eval()
        no_cache_times = measure_inference_time(model, seq_lens, args.d_model, use_kv_cache=False)
        cache_times = measure_inference_time(model, seq_lens, args.d_model, use_kv_cache=True)
        # Plot
        plt.plot(seq_lens, no_cache_times, label="No KV Cache")
        plt.plot(seq_lens, cache_times, label="With KV Cache")
        plt.xlabel("Sequence Length")
        plt.ylabel("Avg Time per Token (s)")
        plt.title("Inference Time vs. Sequence Length")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
