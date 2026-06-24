import modal

app = modal.App("basic-triton-kernel")

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "triton")


@app.function(
    gpu="A10",
    image=image,
    timeout=600,
)
def run_basic_triton():
    import torch
    import triton
    import triton.language as tl

    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch version:", torch.__version__)
    print("Triton Version:", triton.__version__)

    # write kernel - 1 c = a + b

    @triton.jit
    def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        c = a + b

        tl.store(c_ptr + offsets, c, mask=mask)

    def vector_add(a, b):
        assert a.is_cuda and b.is_cuda
        assert a.shape == b.shape

        c = torch.empty_like(a)

        n_elements = a.numel()
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)

        vector_add_kernel[grid](
            a,
            b,
            c,
            n_elements,
            BLOCK_SIZE=block_size,
        )

        return c

    torch.manual_seed(42)

    n = 10_000

    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")

    print("\nInput shape:", a.shape)

    # Test vector add
    c_triton = vector_add(a, b)
    c_torch = a + b

    print("\nVector Add")
    print("Triton output first 5:", c_triton[:5])
    print("Torch output first 5: ", c_torch[:5])
    print("Matches PyTorch:", torch.allclose(c_triton, c_torch, atol=1e-6))


@app.local_entrypoint()
def main():
    run_basic_triton.remote()
