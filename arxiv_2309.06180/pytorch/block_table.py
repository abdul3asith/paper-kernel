import math

import torch


def generate_block_table(seq_lens, block_size, num_blocks):
    """
    seq_lens: list of sequence lengths
              example: [6, 3, 9]

    block_size: tokens per block
                example: 4

    num_blocks: total physical blocks available
                example: 10
    """

    batch_size = len(seq_lens)

    max_seq_len = max(seq_lens)
    max_blocks_per_seq = math.ceil(max_seq_len / block_size)

    block_table = torch.full(
        (batch_size, max_blocks_per_seq),
        fill_value=-1,
        dtype=torch.long,
    )

    free_blocks = list(range(num_blocks))

    for seq_id, seq_len in enumerate(seq_lens):
        blocks_needed = math.ceil(seq_len / block_size)

        for logical_block_id in range(blocks_needed):
            physical_block_id = free_blocks.pop(0)

            block_table[seq_id, logical_block_id] = physical_block_id

    return block_table


if __name__ == "__main__":
    seq_lens = [6, 3, 9]
    block_size = 4
    num_blocks = 10

    block_table = generate_block_table(
        seq_lens=seq_lens,
        block_size=block_size,
        num_blocks=num_blocks,
    )

    print(block_table)
