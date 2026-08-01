# Build Order

## normal attention - 
BATCH
│
├── Sequence 0
│   ├── Token 0
│   │   ├── Head 0 → head_dim vector
│   │   ├── Head 1 → head_dim vector
│   │   └── Head 2 → head_dim vector
│   │
│   ├── Token 1
│   │   ├── Head 0 → head_dim vector
│   │   ├── Head 1 → head_dim vector
│   │   └── Head 2 → head_dim vector
│   │
│   └── Token 2
│       ├── Head 0 → head_dim vector
│       ├── Head 1 → head_dim vector
│       └── Head 2 → head_dim vector
│
├── Sequence 1
│   ├── Token 0
│   ├── Token 1
│   └── Token 2
│
└── Sequence 2
    ├── Token 0
    ├── Token 1
    └── Token 2

attention - 
              K
              |
              v
Q  --------> Q @ Kᵀ --------> scores
                              [batch, heads, q_len, seq_len]
                                      |
                                      v
                                  softmax
                                      |
                                      v
                                  weights
                              [batch, heads, q_len, seq_len]
                                      |
                                      v
weights @ V ---------------------> output
                              [batch, heads, q_len, head_dim]

## contiguous KV cache
## paged KV cache
## block table
## correctness test