import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,  # Pointer to input tensor
    scale_ptr,  # Pointer to scale tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,
    num_features: tl.constexpr,
    eps: tl.constexpr,
):
    # Each program processes a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute mean of x^2 along the feature dimension
    x2 = x * x
    mean = tl.sum(x2, axis=0) / num_features
    # Compute RMS and add epsilon
    rms = tl.sqrt(mean + eps)
    # Compute scale (1 / rms)
    scale = 1.0 / rms
    # Scale the input
    out = x * scale
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_rmsnorm(x: torch.Tensor, eps: float = 1e-5):
    """
    Applies RMS Normalization using a custom Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    num_features = x.size(1)
    n_elements = x.numel()
    scale = torch.rsqrt(torch.mean(x * x, dim=1, keepdim=True) + eps)
    scale = scale.to(x.dtype)
    out = torch.empty_like(x)

    # Tunable block size
    BLOCK_SIZE = 128

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    rmsnorm_kernel[grid](x, scale, out, n_elements, BLOCK_SIZE, num_features, eps)
    return out


class ModelNew(nn.Module):
    """
    Optimized RMS Normalization using custom Triton kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rmsnorm(x, self.eps)