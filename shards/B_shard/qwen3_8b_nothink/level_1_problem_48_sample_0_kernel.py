import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in the input
    dim,  # Dimension to reduce over
    BLOCK_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the offset for the current block
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute the sum along the specified dimension
    sum_val = tl.sum(x, axis=dim)
    # Compute the count of elements along the dimension
    count = tl.arange(0, BLOCK_SIZE)
    count = tl.sum(count, axis=dim)
    # Compute the mean
    mean = sum_val / count
    # Store the result
    tl.store(out_ptr + offsets, mean, mask=mask)


def triton_mean_reduction(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    mean_reduction_kernel[grid](x, out, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE, STRIDE=1)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension using a custom Triton kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        return triton_mean_reduction(x, self.dim)