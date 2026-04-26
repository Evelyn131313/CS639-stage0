import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    input_ptr,  # Pointer to input tensor
    mean_ptr,   # Pointer to mean tensor
    var_ptr,    # Pointer to variance tensor
    output_ptr, # Pointer to output tensor
    batch_size: tl.constexpr,
    num_features: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the offset for the current program
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the index in the input tensor
    idx = offset
    # Compute the index in the feature dimension
    feature_idx = idx % num_features
    # Compute the index in the spatial dimensions
    spatial_idx = idx // num_features
    # Compute the index in the batch dimension
    batch_idx = spatial_idx // (height * width)
    # Compute the index in the spatial dimensions (flattened)
    spatial_flat = spatial_idx % (height * width)
    # Compute the index in the input tensor
    input_idx = batch_idx * num_features * height * width + feature_idx * height * width + spatial_flat
    # Load input value
    x = tl.load(input_ptr + input_idx, mask=offset < batch_size * num_features * height * width, other=0.0)
    # Compute mean and variance
    mean = tl.load(mean_ptr + feature_idx, other=0.0)
    var = tl.load(var_ptr + feature_idx, other=0.0)
    # Compute normalization
    inv_std = tl.rsqrt(var + eps)
    # Compute output
    out = (x - mean) * inv_std
    # Store output
    tl.store(output_ptr + input_idx, out, mask=offset < batch_size * num_features * height * width)


def triton_instance_norm(input: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, eps: float = 1e-5):
    """
    Applies Instance Normalization using a custom Triton kernel.
    """
    assert input.is_cuda and mean.is_cuda and var.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()

    # Output tensor
    output = torch.empty_like(input)

    # Constants
    batch_size = input.size(0)
    num_features = input.size(1)
    height = input.size(2)
    width = input.size(3)

    # Block size
    BLOCK_SIZE = 128

    # Grid size
    grid = lambda meta: ((batch_size * num_features + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    instance_norm_kernel[grid](input, mean, var, output, batch_size, num_features, height, width, eps, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using custom Triton kernels.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.register_buffer('mean', torch.zeros(num_features))
        self.register_buffer('var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean and variance across spatial dimensions
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        # Flatten mean and var to match feature dimension
        mean = mean.view(-1, self.num_features)
        var = var.view(-1, self.num_features)
        # Apply Instance Normalization using Triton kernel
        return triton_instance_norm(x, mean, var, eps=1e-5)