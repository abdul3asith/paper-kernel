import math

import torch
from contiguous_kv_cache import ContiguousKVCache, normal_attention_single_seq
from paged_kv_cache import PagedKVCache


def test_paged_matches_contiguous():
    torch.manual_seed(42)

    batch_size = 3
    num_heads = 4
    max_seq_len = 10
    head_dim = 8

    block_size = 4
    num_blocks = 20

    seq_lens = [6, 3, 9]

    contiguous_cache = ContiguousKVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
    )

    paged_cache = PagedKVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=num_blocks,
    )

    # Add the exact same K/V tokens to both caches
    for batch_idx, seq_len in enumerate(seq_lens):
        for _ in range(seq_len):
            new_k = torch.randn(num_heads, head_dim)
            new_v = torch.randn(num_heads, head_dim)

            contiguous_cache.append(batch_idx, new_k, new_v)
            paged_cache.append(batch_idx, new_k, new_v)

    print("Contiguous seq_lens:", contiguous_cache.seq_lens)
    print("Paged seq_lens:     ", paged_cache.seq_lens)

    assert torch.equal(contiguous_cache.seq_lens, paged_cache.seq_lens)

    print("\nPaged block table:")
    print(paged_cache.block_table)

    print("\nPaged memory stats:")
    print(paged_cache.memory_stats())

    # Compare each sequence
    for batch_idx in range(batch_size):
        print(f"\nChecking sequence {batch_idx}")

        contig_k, contig_v = contiguous_cache.get_kv(batch_idx)
        paged_k, paged_v = paged_cache.get_kv(batch_idx)

        print("contig_k shape:", contig_k.shape)
        print("paged_k shape: ", paged_k.shape)

        # 1. K/V reconstruction correctness
        assert torch.allclose(contig_k, paged_k, atol=1e-6)
        assert torch.allclose(contig_v, paged_v, atol=1e-6)

        print("K/V match: True")

        # Same query for both attention paths
        q = torch.randn(num_heads, 1, head_dim)

        contig_out, contig_weights = normal_attention_single_seq(
            q,
            contig_k,
            contig_v,
        )

        paged_out, paged_weights = normal_attention_single_seq(
            q,
            paged_k,
            paged_v,
        )

        # 2. Attention output correctness
        assert torch.allclose(contig_out, paged_out, atol=1e-6)
        assert torch.allclose(contig_weights, paged_weights, atol=1e-6)

        print("Attention output match: True")
        print("Attention weights match: True")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    test_paged_matches_contiguous()
