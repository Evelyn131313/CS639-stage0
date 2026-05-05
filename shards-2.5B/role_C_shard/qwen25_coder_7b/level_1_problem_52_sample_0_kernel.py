import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    dim_size,  # Size of the dimension to perform argmin on
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Initialize argmin and its index
    argmin_val = tl.max_s32(tl.float32(-1e30))
    argmin_index = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    # Iterate over the dimension to find argmin
    for i in range(dim_size):
        current_val = x[i]
        current_index = offsets + i
        current_mask = current_val < argmin_val
        argmin_val = tl.where(current_mask, current_val, argmin_val)
        argmin_index = tl.where(current_mask, current_index, argmin_index)
    # Store the result
    tl.store(out_ptr + offsets, argmin_index, mask=mask)


def triton_argmin(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x, dtype=torch.int32)

    # Number of elements in the tensor
    n_elements = x.numel()
    dim_size = x.shape[dim]
    BLOCK_SIZE = 256  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    argmin_kernel[grid](x, out, n_elements, dim_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Simple model that finds the index of the minimum value along a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        return triton_argmin(x, self.dim)