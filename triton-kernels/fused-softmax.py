"""
The softmax function transforms a vector of raw, unnormalized scores (called logits) into a probability
distribution, where each output is a value between \(0\) and \(1\), and all values sum to \(1\)
"""

# softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
# x_max = max.x(dim=1)[0]
# z = x - x_max
# numerator = exp(z)[:, None]
# denominator = numerator.sum(dim=1)
# sofmax = numerator/denominato[:, None]
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
def run_fused_softmax():
    import torch
    import triton
    import triton.language as tl
    from triton.runtime import driver

    print("CUDA available:", torch.cuda.is_available())
    print("HIP/ROCm build:", torch.version.hip)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch version:", torch.__version__)
    print("Triton version:", triton.__version__)

    def naive_softmax(x):

        # read  MN elements ; write M  elements
        x_max = x.max(dim=1)[0]
        # read MN + M elements ; write MN elements
        z = x - x_max[:, None]
        # read  MN elements ; write MN elements
        numerator = torch.exp(z)
        # read  MN elements ; write M  elements
        denominator = numerator.sum(dim=1)
        # read MN + M elements ; write MN elements
        ret = numerator / denominator[:, None]
        # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
        return ret

    @triton.jit  # this line compiles below function into a gpu kernel
    def softmax_kernel(
        output_ptr,
        input_ptr,
        input_row_stride,
        output_row_stride,
        n_rows,
        n_cols,
        BLOCK_SIZE: tl.contexpr,
        num_stages: tl.contexpr,
    ):
        row_start = tl.program_id(0)
        row_step = tl.num_programs(0)

        # stride of 0, 4, 8, --> 1, 5, 9,

        for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
            row_start_ptr = input_ptr + row_idx * input_row_stride

            col_offsets = tl.arange(0, BLOCK_SIZE)
            input_ptrs = row_start_ptr + col_offsets

            mask = col_offsets < n_cols

            row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

            row_minus_max = row - tl.max(row, axis=0)

            numerator = tl.exp(row_minus_max)
            denominator = tl.sum(numerator, axis=0)

            softmax_output = numerator / denominator

            output_row_start_ptr = output_ptr + row_idx * output_row_stride
            output_ptrs = output_row_start_ptr + col_offsets
            tl.store(output_ptrs, softmax_output, mask=mask)


@app.local_entrypoint()
def main():
    run_fused_softmax.remote()
