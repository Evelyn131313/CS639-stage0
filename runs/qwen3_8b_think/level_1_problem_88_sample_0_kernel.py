import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return triton_gelu(x)

@triton.jit
def gelu_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Compute block start and offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute x^3
    x_cubed = x * x * x
    # Compute 0.044715 * x^3
    term = 0.044715 * x_cubed
    # Add x
    term_plus_x = x + term
    # Multiply by sqrt(2/pi)
    sqrt_2_over_pi = 0.7978845608
    arg = sqrt_2_over_pi * term_plus_x
    # Apply tanh
    tanh_arg = tl.tanh(arg)
    # Compute final result
    result = 0.5 * x * (1.0 + tanh_arg)
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

def triton_gelu(x: torch.Tensor):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out