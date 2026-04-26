import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements in the row
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread processes one element in the row
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    # Assume that the block is responsible for a single row, and the threads process the elements in the row
    # Shared memory is used to store the cumulative sums
    shared = tl.shared(n_elements)
    # Load the element from the input
    x = tl.load(x_ptr + tid, mask=..., other=0.0)
    # Compute the cumulative sum
    if tid == 0:
        shared[tid] = x
    else:
        shared[tid] = shared[tid-1] + x
    # Synchronize to ensure that all threads have completed their computation
    tl.sync()
    # Write the result to the output
    tl.store(out_ptr + tid, shared[tid])


def triton_cumsum(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    cumsum_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Compute cumulative sum along the specified dimension using Triton kernel
        return triton_cumsum(x, self.dim)