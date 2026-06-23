import math

import torch
import torch.nn.functional as F


def normal_attention(q, k_cache, v_cache):
    batch, heads, q_len, head_dim = q.shape
    _, _, seq_len, _ = k_cache.shape

    assert q_len == 1  # "For now we only support one decode token"
    assert k_cache.shape == v_cache.shape
    assert k_cache.shape[0] == batch
    assert k_cache.shape[1] == heads
    assert k_cache.shape[3] == head_dim

    # normalization = softmax(qk^t / sqrt(d)).v

    scores = torch.matmul(q, k_cache.transpose(-2, -1))

    # Step 2: scale by sqrt(head_dim)
    scores = scores / math.sqrt(head_dim)

    # Step 3: softmax over sequence dimension
    # weights: [B, H, 1, S]
    weights = torch.softmax(scores, dim=-1)

    # Step 4: multiply by values
    # weights: [B, H, 1, S]
    # v_cache: [B, H, S, D]
    # out:     [B, H, 1, D]
    out = torch.matmul(weights, v_cache)

    return out, weights


if __name__ == "__main__":
    torch.manual_seed(42)

    batch = 2
    heads = 4
    seq_len = 6
    head_dim = 8

    q = torch.randn(batch, heads, 1, head_dim)
    k_cache = torch.randn(batch, heads, seq_len, head_dim)
    v_cache = torch.randn(batch, heads, seq_len, head_dim)

    out, weights = normal_attention(q, k_cache, v_cache)

    print("q shape:", q.shape)
    print("k_cache shape:", k_cache.shape)
    print("v_cache shape:", v_cache.shape)
    print("attention weights shape:", weights.shape)
    print("output shape:", out.shape)

    # Check that attention weights sum to 1 over seq_len
    print("weights sum:", weights.sum(dim=-1))

    # Compare with PyTorch built-in attention
    torch_out = F.scaled_dot_product_attention(
        q,
        k_cache,
        v_cache,
        is_causal=False,
    )

    print("Matches PyTorch:", torch.allclose(out, torch_out, atol=1e-6))
