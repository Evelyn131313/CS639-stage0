import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_kernel(
    x_ptr,  # Pointer to input tensor
    mask_ptr,  # Pointer to boolean mask
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mask = tl.load(mask_ptr + offsets, mask=mask, other=False)
    
    # Initialize cumulative sum
    cumsum = tl.zeros_like(x)
    
    # Perform the masked cumulative sum
    for i in range(dim):
        if i < dim - 1:
            tl.store(out_ptr + offsets, cumsum, mask=mask)
            cumsum = tl.where(mask, cumsum + x, cumsum)
        else:
            tl.store(out_ptr + offsets, cumsum + x, mask=mask)
    
    # Store the final result
    tl.store(out_ptr + offsets, cumsum, mask=mask)


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    mask = mask.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 256  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    masked_cumsum_kernel[grid](x, mask, out, n_elements, dim=x.dim(), BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        return triton_masked_cumsum(x, mask)