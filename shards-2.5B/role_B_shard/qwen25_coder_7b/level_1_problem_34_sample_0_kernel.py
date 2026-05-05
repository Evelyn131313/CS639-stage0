import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr,  # Pointer to input tensor
    mean_ptr,  # Pointer to mean tensor
    var_ptr,  # Pointer to variance tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    eps,  # Small constant to prevent division by zero
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
    # Load mean and variance
    mean = tl.load(mean_ptr, mask=mask, other=0.0)
    var = tl.load(var_ptr, mask=mask, other=0.0)
    # Compute the normalized value
    x_norm = (x - mean) / tl.sqrt(var + eps)
    # Store the result
    tl.store(out_ptr + offsets, x_norm, mask=mask)


def triton_instance_norm(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, eps: float):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and mean.is_cuda and var.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    instance_norm_kernel[grid](x, mean, var, out, n_elements, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs Instance Normalization using Triton kernels.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = 1e-5
        self.register_buffer('mean', torch.zeros(num_features))
        self.register_buffer('var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        # Flatten the spatial dimensions to perform Instance Norm on the feature dimension
        batch_size, features, height, width = x.shape
        x_flat = x.view(batch_size, features, -1)
        x_flat = x_flat.transpose(1, 2).contiguous()

        # Compute mean and variance
        mean = x_flat.mean(dim=-1, keepdim=True)
        var = x_flat.var(dim=-1, keepdim=True)

        # Update running mean and variance (not shown here for simplicity)
        self.mean = self.mean * 0.9 + mean * 0.1
        self.var = self.var * 0.9 + var * 0.1

        # Apply Instance Normalization using Triton kernel
        out_flat = triton_instance_norm(x_flat, mean, var, self.eps)
        out = out_flat.transpose(1, 2).contiguous().view(batch_size, features, height, width)
        return out