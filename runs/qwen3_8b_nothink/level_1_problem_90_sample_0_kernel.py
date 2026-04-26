import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in the tensor
    dim,  # Dimension along which to compute the cumulative product
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the index in the batch
    pid = tl.program_id(0)
    # Compute the offset in the batch
    batch_offset = pid * BLOCK_SIZE
    # Compute the offset in the dimension
    dim_offset = tl.arange(0, BLOCK_SIZE)
    # Compute the offset in the rest of the dimensions
    rest_offset = tl.arange(0, BLOCK_SIZE)
    # Compute the total offset
    offsets = batch_offset + dim_offset * (n_elements // BLOCK_SIZE) + rest_offset
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute cumulative product
    out = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        if i == 0:
            out[i] = x[i]
        else:
            out[i] = out[i - 1] * x[i]
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_cumprod(x: torch.Tensor, dim: int):
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
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    cumprod_kernel[grid](x, out, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension,
    optimized with a custom Triton kernel.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return triton_cumprod(x, dim=self.dim)