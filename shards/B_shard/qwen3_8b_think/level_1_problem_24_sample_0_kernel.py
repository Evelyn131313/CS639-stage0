import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    dim: tl.constexpr,
    batch_size: tl.constexpr,
    num_elements_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a row
    row_idx = tl.program_id(0)
    # Compute the offset for the row
    row_start = row_idx * num_elements_per_row
    # Create a range of offsets for the row
    offsets = row_start + tl.arange(0, num_elements_per_row)
    # Mask to ensure we don't go out of bounds
    mask = offsets < (row_start + num_elements_per_row)
    # Load the input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute the max of the row
    max_val = tl.max(x)
    # Subtract the max from each element
    x_sub = x - max_val
    # Compute the exponentials
    exp_x_sub = tl.exp(x_sub)
    # Compute the sum of exponentials
    sum_exp = tl.sum(exp_x_sub)
    # Compute the log of the sum
    log_sum_exp = tl.log(sum_exp)
    # Compute the log_softmax
    log_softmax = x_sub - log_sum_exp
    # Store the result
    tl.store(out_ptr + offsets, log_softmax, mask=mask)

def triton_log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size = x.size(0)
    num_elements_per_row = x.size(1)
    # Set the block size to the number of elements per row
    BLOCK_SIZE = num_elements_per_row
    # Determine the grid size (number of rows)
    grid = (batch_size,)
    # Launch the kernel
    log_softmax_kernel[grid](x, out, dim, batch_size, num_elements_per_row, BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_log_softmax(x, self.dim)