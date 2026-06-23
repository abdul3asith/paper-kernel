import math

import torch


class PagedKVCache:
    def __init__(
        self, batch_size, num_heads, max_seq_len, head_dim, block_size, num_blocks
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Physical KV memory
        # Shape follows PagedAttention idea:
        # [physical_blocks, tokens_per_block, heads, head_dim]
        self.key_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)
        self.value_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)

        # Max number of blocks one sequence may need
        self.max_blocks_per_seq = math.ceil(max_seq_len / block_size)

        # block_table[batch_idx, logical_block_idx] = physical_block_id
        self.block_table = torch.full(
            (batch_size, self.max_blocks_per_seq),
            fill_value=-1,
            dtype=torch.long,
        )

        # Current number of tokens in each sequence
        self.seq_lens = torch.zeros(batch_size, dtype=torch.long)

        # Free physical blocks
        self.free_blocks = list(range(num_blocks))

    def allocate_block(self, batch_idx, logical_block_idx):
        """
        Allocates one physical block for a sequence.
        """

        if self.block_table[batch_idx, logical_block_idx] != -1:
            return self.block_table[batch_idx, logical_block_idx].item()

        if len(self.free_blocks) == 0:
            raise RuntimeError("No free KV blocks left")

        physical_block_id = self.free_blocks.pop(0)
        self.block_table[batch_idx, logical_block_idx] = physical_block_id

        return physical_block_id

    def append(self, batch_idx, new_k, new_v):
        """
        Adds one new token's K/V to the paged cache.

        new_k: [heads, head_dim]
        new_v: [heads, head_dim]
        """

        assert new_k.shape == (self.num_heads, self.head_dim)
        assert new_v.shape == (self.num_heads, self.head_dim)

        pos = self.seq_lens[batch_idx].item()

        if pos >= self.max_seq_len:
            raise RuntimeError("Sequence reached max_seq_len")

        # Logical position inside this sequence
        logical_block_idx = pos // self.block_size
        offset_inside_block = pos % self.block_size

        # Find or allocate physical block
        physical_block_id = self.allocate_block(batch_idx, logical_block_idx)

        # Store token K/V inside physical memory
        self.key_cache[physical_block_id, offset_inside_block, :, :] = new_k
        self.value_cache[physical_block_id, offset_inside_block, :, :] = new_v

        self.seq_lens[batch_idx] += 1

    def get_kv(self, batch_idx):
        """
        Reconstructs logical K/V for one sequence by following block_table.

        returns:
        k: [heads, seq_len, head_dim]
        v: [heads, seq_len, head_dim]
        """

        seq_len = self.seq_lens[batch_idx].item()

        if seq_len == 0:
            raise RuntimeError("Sequence has no cached tokens")

        num_used_blocks = math.ceil(seq_len / self.block_size)

        physical_blocks = self.block_table[batch_idx, :num_used_blocks]

        # Gather physical blocks
        # [num_used_blocks, block_size, heads, head_dim]
        k_blocks = self.key_cache[physical_blocks]
        v_blocks = self.value_cache[physical_blocks]

        # Flatten blocks into logical token order
        # [num_used_blocks * block_size, heads, head_dim]
        k_flat = k_blocks.reshape(-1, self.num_heads, self.head_dim)
        v_flat = v_blocks.reshape(-1, self.num_heads, self.head_dim)

        # Remove unused padding slots from the last block
        # [seq_len, heads, head_dim]
        k_flat = k_flat[:seq_len]
        v_flat = v_flat[:seq_len]

        # Convert to attention format:
        # [heads, seq_len, head_dim]
        k = k_flat.permute(1, 0, 2)
        v = v_flat.permute(1, 0, 2)

        return k, v

    def free_sequence(self, batch_idx):
        """
        Frees all physical blocks used by one sequence.
        """

        seq_len = self.seq_lens[batch_idx].item()

        if seq_len == 0:
            return

        num_used_blocks = math.ceil(seq_len / self.block_size)

        for logical_block_idx in range(num_used_blocks):
            physical_block_id = self.block_table[batch_idx, logical_block_idx].item()

            if physical_block_id != -1:
                self.free_blocks.append(physical_block_id)
                self.block_table[batch_idx, logical_block_idx] = -1

        self.seq_lens[batch_idx] = 0

    def memory_stats(self):
        """
        Shows how much block memory is used/wasted.
        """

        allocated_blocks = (self.block_table != -1).sum().item()
        used_tokens = self.seq_lens.sum().item()

        allocated_token_slots = allocated_blocks * self.block_size
        wasted_token_slots = allocated_token_slots - used_tokens

        return {
            "allocated_blocks": allocated_blocks,
            "used_tokens": used_tokens,
            "allocated_token_slots": allocated_token_slots,
            "wasted_token_slots": wasted_token_slots,
            "free_blocks": len(self.free_blocks),
        }


def normal_attention_single_seq(q, k, v):
    """
    q: [heads, 1, head_dim]
    k: [heads, seq_len, head_dim]
    v: [heads, seq_len, head_dim]

    returns:
    out: [heads, 1, head_dim]
    weights: [heads, 1, seq_len]
    """

    head_dim = q.shape[-1]

    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(head_dim)

    weights = torch.softmax(scores, dim=-1)

    out = torch.matmul(weights, v)

    return out, weights


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    num_heads = 4
    max_seq_len = 10
    head_dim = 8

    block_size = 4
    num_blocks = 8

    cache = PagedKVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=num_blocks,
    )

    # Store original tokens so we can verify correctness
    baseline_k_tokens = {0: [], 1: []}
    baseline_v_tokens = {0: [], 1: []}

    # Add 6 tokens to sequence 0
    for _ in range(6):
        new_k = torch.randn(num_heads, head_dim)
        new_v = torch.randn(num_heads, head_dim)

        cache.append(batch_idx=0, new_k=new_k, new_v=new_v)

        baseline_k_tokens[0].append(new_k)
        baseline_v_tokens[0].append(new_v)

    # Add 3 tokens to sequence 1
    for _ in range(3):
        new_k = torch.randn(num_heads, head_dim)
        new_v = torch.randn(num_heads, head_dim)

        cache.append(batch_idx=1, new_k=new_k, new_v=new_v)

        baseline_k_tokens[1].append(new_k)
        baseline_v_tokens[1].append(new_v)

    print("key_cache shape:", cache.key_cache.shape)
    print("value_cache shape:", cache.value_cache.shape)
    print("block_table:")
    print(cache.block_table)
    print("seq_lens:", cache.seq_lens)
    print("memory stats:", cache.memory_stats())

    # Get paged K/V for sequence 0
    paged_k, paged_v = cache.get_kv(batch_idx=0)

    # Build baseline contiguous K/V for sequence 0
    # list of [heads, head_dim] -> [seq_len, heads, head_dim]
    baseline_k = torch.stack(baseline_k_tokens[0], dim=0)
    baseline_v = torch.stack(baseline_v_tokens[0], dim=0)

    # [seq_len, heads, head_dim] -> [heads, seq_len, head_dim]
    baseline_k = baseline_k.permute(1, 0, 2)
    baseline_v = baseline_v.permute(1, 0, 2)

    print("paged_k shape:", paged_k.shape)
    print("paged_v shape:", paged_v.shape)

    print("K matches baseline:", torch.allclose(paged_k, baseline_k))
    print("V matches baseline:", torch.allclose(paged_v, baseline_v))

    # Test attention correctness
    q = torch.randn(num_heads, 1, head_dim)

    paged_out, paged_weights = normal_attention_single_seq(q, paged_k, paged_v)
    baseline_out, baseline_weights = normal_attention_single_seq(
        q, baseline_k, baseline_v
    )

    print("paged_out shape:", paged_out.shape)
    print("paged_weights shape:", paged_weights.shape)

    print(
        "Attention output matches baseline:",
        torch.allclose(paged_out, baseline_out, atol=1e-6),
    )
    print(
        "Attention weights match baseline:",
        torch.allclose(paged_weights, baseline_weights, atol=1e-6),
    )
