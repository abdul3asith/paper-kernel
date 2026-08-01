import math

import torch


class ContiguousKVCache:
    def __init__(self, batch_size, num_heads, max_seq_len, head_dim):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim

        # Full memory allocated upfront
        self.k_cache = torch.zeros(
            batch_size,
            num_heads,
            max_seq_len,
            head_dim,
        )

        self.v_cache = torch.zeros(
            batch_size,
            num_heads,
            max_seq_len,
            head_dim,
        )

        self.seq_lens = torch.zeros(batch_size, dtype=torch.long)

    def append(self, batch_idx, new_k, new_v):

        pos = self.seq_lens[batch_idx].item()

        if pos >= self.max_seq_len:
            raise ValueError("KV cache is full")

        self.k_cache[batch_idx, :, pos, :] = new_k
        self.v_cache[batch_idx, :, pos, :] = new_v

        self.seq_lens[batch_idx] += 1

    def get_kv(self, batch_idx):

        seq_len = self.seq_lens[batch_idx].item()

        k = self.k_cache[batch_idx, :, :seq_len, :]
        v = self.v_cache[batch_idx, :, :seq_len, :]

        return k, v


def normal_attention_single_seq(q, k, v):
    """
    q: [heads, 1, head_dim]
    k: [heads, seq_len, head_dim]
    v: [heads, seq_len, head_dim]

    out: [heads, 1, head_dim]
    """

    head_dim = q.shape[-1]

    # [heads, 1, head_dim] @ [heads, head_dim, seq_len]
    # -> [heads, 1, seq_len]
    scores = torch.matmul(q, k.transpose(-2, -1))

    scores = scores / math.sqrt(head_dim)

    weights = torch.softmax(scores, dim=-1)

    # [heads, 1, seq_len] @ [heads, seq_len, head_dim]
    # -> [heads, 1, head_dim]
    out = torch.matmul(weights, v)

    return out, weights


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    num_heads = 4
    max_seq_len = 8
    head_dim = 16

    cache = ContiguousKVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
    )

    # Add 3 tokens to sequence 0
    for _ in range(3):
        new_k = torch.randn(num_heads, head_dim)
        new_v = torch.randn(num_heads, head_dim)
        cache.append(batch_idx=0, new_k=new_k, new_v=new_v)

    # Add 5 tokens to sequence 1
    for _ in range(5):
        new_k = torch.randn(num_heads, head_dim)
        new_v = torch.randn(num_heads, head_dim)
        cache.append(batch_idx=1, new_k=new_k, new_v=new_v)

    print("k_cache shape:", cache.k_cache.shape)
    print("v_cache shape:", cache.v_cache.shape)
    print("seq_lens:", cache.seq_lens)

    # Now decode one new token for sequence 0
    q = torch.randn(num_heads, 1, head_dim)

    k, v = cache.get_kv(batch_idx=0)

    out, weights = normal_attention_single_seq(q, k, v)

    print("q shape:", q.shape)
    print("k shape:", k.shape)
    print("v shape:", v.shape)
    print("weights shape:", weights.shape)
    print("out shape:", out.shape)
