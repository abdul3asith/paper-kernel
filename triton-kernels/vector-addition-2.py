import modal

app = modal.App("triton-vector-add-matplotlib")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "triton",
    "matplotlib",
)


@app.function(
    gpu="A10",
    image=image,
    timeout=600,
)
def run_vector_add_tutorial():
    import io

    import matplotlib
    import torch
    import triton
    import triton.language as tl

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch version:", torch.__version__)
    print("Triton version:", triton.__version__)

    @triton.jit
    def add_kernel(
        x_ptr,
        y_ptr,
        z_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Which Triton program/block is this?
        pid = tl.program_id(axis=0)

        # Start index for this block
        block_start = pid * BLOCK_SIZE

        # Elements handled by this block
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        # Avoid reading/writing past the end
        mask = offsets < n_elements

        # Load from GPU memory
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        z = tl.load(z_ptr + offsets, mask=mask)

        # Compute
        output = x + y + z

        # Store to GPU memory
        tl.store(output_ptr + offsets, output, mask=mask)

    def add(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        output = torch.empty_like(x)

        n_elements = output.numel()
        block_size = 1024

        grid = (triton.cdiv(n_elements, block_size),)

        add_kernel[grid](
            x,
            y,
            z,
            output,
            n_elements,
            BLOCK_SIZE=block_size,
        )

        return output

    # ------------------------------------------------------------
    # Correctness test
    # ------------------------------------------------------------
    torch.manual_seed(0)

    size = 98_432

    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    z = torch.rand(size, device="cuda")

    output_torch = x + y + z
    output_triton = add(x, y, z)

    torch.cuda.synchronize()

    max_diff = torch.max(torch.abs(output_torch - output_triton)).item()

    print("\nCorrectness test")
    print("Torch first 5: ", output_torch[:5])
    print("Triton first 5:", output_triton[:5])
    print("Max difference:", max_diff)
    print("Matches:", torch.allclose(output_torch, output_triton, atol=1e-6))

    # ------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------
    sizes = [2**i for i in range(12, 27)]  # 4096 to 67M
    triton_gbps = []
    torch_gbps = []

    def gbps(num_elements, ms):
        # 3 tensors touched: x read, y read, output write
        bytes_processed = 3 * num_elements * 4
        seconds = ms * 1e-3
        return bytes_processed / seconds / 1e9

    for n in sizes:
        x = torch.rand(n, device="cuda", dtype=torch.float32)
        y = torch.rand(n, device="cuda", dtype=torch.float32)
        z = torch.rand(n, device="cuda", dtype=torch.float32)

        torch_ms = triton.testing.do_bench(lambda: x + y + z)
        triton_ms = triton.testing.do_bench(lambda: add(x, y, z))

        torch_speed = gbps(n, torch_ms)
        triton_speed = gbps(n, triton_ms)

        torch_gbps.append(torch_speed)
        triton_gbps.append(triton_speed)

        print(
            f"size={n:<10} "
            f"torch={torch_speed:>8.2f} GB/s  "
            f"triton={triton_speed:>8.2f} GB/s"
        )

    # ------------------------------------------------------------
    # Matplotlib plot
    # ------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    plt.plot(sizes, torch_gbps, marker="o", label="PyTorch")
    plt.plot(sizes, triton_gbps, marker="o", label="Triton")

    plt.xscale("log", base=2)
    plt.xlabel("Vector size")
    plt.ylabel("Bandwidth GB/s")
    plt.title("Vector Add Performance: PyTorch vs Triton")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    buffer.seek(0)

    return buffer.getvalue()


@app.local_entrypoint()
def main():
    plot_bytes = run_vector_add_tutorial.remote()

    output_path = "vector_add_performance.png"

    with open(output_path, "wb") as f:
        f.write(plot_bytes)

    print(f"\nSaved plot locally as: {output_path}")
