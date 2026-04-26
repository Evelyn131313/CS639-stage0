import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def selu_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
    alpha: tl.constexpr,
    scale: tl.constexpr
):
    # Each program processes a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute SELU: out = alpha * (max(0, x) + scale * x)
    # First compute max(0, x)
    pos = tl.maximum(x, 0.0)
    # Then compute scale * x
    scaled = tl.load(x_ptr + offsets, mask=mask, other=0.0) * scale
    # Add them
    out = alpha * (pos + scaled)
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_selu(x: torch.Tensor):
    """
    Applies SELU activation using a custom Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # SELU parameters
    alpha = 1.673263242354177
    scale = 1.0507009873586021

    selu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, alpha=alpha, scale=scale)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_selu(x)