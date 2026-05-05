import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    dim_size,  # Size of the dimension along which to perform the operation
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < dim_size
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Perform the reverse cumulative sum
    cumsum = tl.zeros_like(x)
    for i in range(1, dim_size):
        cumsum = x[i] + cumsum
    # Store the result
    tl.store(out_ptr + offsets, cumsum, mask=mask)


def triton_reverse_cumsum(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    dim_size = x.size(dim)
    BLOCK_SIZE = 256  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((dim_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    reverse_cumsum_kernel[grid](x, out, n_elements, dim_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Perform the reverse cumulative sum along the specified dimension
        return triton_reverse_cumsum(x, self.dim)