import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def min_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_batch,  # Number of batches
    n_dim1,  # Size of dim1
    n_dim2,  # Size of dim2
    dim: tl.constexpr,  # Dimension to reduce
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // n_dim2
    i = pid % n_dim2

    # Compute the starting index in the input tensor for (b, i)
    if dim == 1:
        # Reduce along dim1 (second dimension)
        # Each (b, i) in output corresponds to (b, :, i) in input
        start_idx = b * n_dim1 * n_dim2 + i
        # Initialize min_val to +infinity
        min_val = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32)
        # Compute the number of blocks
        num_blocks = (n_dim1 + BLOCK_SIZE - 1) // BLOCK_SIZE
        # For each block
        for block_idx in range(num_blocks):
            # Compute the starting k for this block
            start_k = block_idx * BLOCK_SIZE
            end_k = start_k + BLOCK_SIZE
            end_k = tl.minimum(end_k, n_dim1)
            # Compute the offset for this block
            offset = start_k * n_dim2
            # Load the elements in this block
            x_block = tl.load(x_ptr + start_idx + offset, mask=offset < n_dim1 * n_dim2, other=0.0)
            # Compute the min of the block
            min_block = tl.min(x_block)
            # Update the overall min_val
            min_val = tl.min(min_val, min_block)
        # Store the final min_val
        out_idx = b * n_dim2 + i
        tl.store(out_ptr + out_idx, min_val)
    elif dim == 2:
        # Reduce along dim2 (third dimension)
        # Each (b, i) in output corresponds to (b, i, :) in input
        start_idx = b * n_dim1 * n_dim2 + i * n_dim1
        # Initialize min_val to +infinity
        min_val = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32)
        # Compute the number of blocks
        num_blocks = (n_dim2 + BLOCK_SIZE - 1) // BLOCK_SIZE
        # For each block
        for block_idx in range(num_blocks):
            # Compute the starting k for this block
            start_k = block_idx * BLOCK_SIZE
            end_k = start_k + BLOCK_SIZE
            end_k = tl.minimum(end_k, n_dim2)
            # Compute the offset for this block
            offset = start_k
            # Load the elements in this block
            x_block = tl.load(x_ptr + start_idx + offset, mask=offset < n_dim2, other=0.0)
            # Compute the min of the block
            min_block = tl.min(x_block)
            # Update the overall min_val
            min_val = tl.min(min_val, min_block)
        # Store the final min_val
        out_idx = b * n_dim1 + i
        tl.store(out_ptr + out_idx, min_val)

def triton_min_reduction(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    if dim == 1:
        output_shape = (x.size(0), x.size(2))
    elif dim == 2:
        output_shape = (x.size(0), x.size(1))
    else:
        raise ValueError("Unsupported dimension for reduction.")
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    n_batch = x.size(0)
    n_dim1 = x.size(1)
    n_dim2 = x.size(2)
    BLOCK_SIZE = 128
    # Determine the grid size
    if dim == 1:
        grid = (n_batch * n_dim2,)
    elif dim == 2:
        grid = (n_batch * n_dim1,)
    else:
        raise ValueError("Unsupported dimension for reduction.")
    # Launch the kernel
    min_reduction_kernel[grid](x, out, n_batch, n_dim1, n_dim2, dim, BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_min_reduction(x, self.dim)