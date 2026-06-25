



# PaperKernel

A monorepo for implementing influential research papers from scratch.

The goal of this repo is to turn important AI, ML systems, and GPU research papers into readable, working code. Each implementation is built step by step to understand the core ideas, verify correctness, and explore performance trade-offs.

## Focus Areas

This repo will mainly cover papers related to:

* Deep learning
* Transformers
* LLM inference
* Attention mechanisms
* GPU kernels
* Memory optimization
* Distributed systems
* Model serving
* ML systems

## Implementation Approach

Most papers will follow this structure:

```text
1. PyTorch reference implementation
2. Triton implementation, when useful
3. CUDA implementation, when useful
4. Tests and benchmarks
5. Notes and explanations
```

## Repo Structure


paperkernel/
  README.md
  requirements.txt

  papers/
    paper_name/
      README.md

      pytorch/
      triton/
      cuda/
      tests/
      benchmarks/
      notes/
      results/

## Goals

* Understand research papers by implementing them
* Build simple and readable reference versions
* Compare different implementation approaches
* Learn GPU programming and ML systems deeply
* Document mistakes, trade-offs, and benchmarks
* Create a strong public portfolio of research-to-code work

## Not Goals

This repo is not meant to be a production library or a replacement for optimized frameworks.

The priority is learning, correctness, clarity, and engineering depth.

## Planned Topics

* Attention mechanisms
* KV-cache optimization
* FlashAttention-style kernels
* PagedAttention-style memory management
* Speculative decoding
* Quantization
* LoRA / QLoRA
* Mixture of Experts
* Transformer architectures
* Distributed inference
* GPU kernel optimization

## Status

This repo is actively being built.

Each paper will have its own folder with implementation details, notes, tests, and benchmarks.

## 1. https://arxiv.org/pdf/2309.06180 - Efficient Memory Management in LLM Serving (Paged Attention)

## Author

Built by Abdul Basith as a public learning project in GPU programming, ML systems, and research paper implementation.
