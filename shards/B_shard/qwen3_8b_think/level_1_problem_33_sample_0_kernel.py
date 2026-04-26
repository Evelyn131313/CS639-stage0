import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batchnorm_kernel(
    x_ptr,  # Pointer to input tensor
    y_ptr,  # Pointer to output tensor
    gamma_ptr,  # Pointer to gamma parameter
    beta_ptr,  # Pointer to beta parameter
    mean_ptr,  # Pointer to mean tensor
    var_ptr,  # Pointer to variance tensor
    num_channels: tl.constexpr,
    batch_size: tl.constexpr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread processes one element
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < x_ptr.size

    # Compute the channel index for each element
    # Assuming the input tensor is stored in (batch, channel, height, width) order
    # So for a given offset, the channel index is (offset // (batch_size * dim1 * dim2))
    channel_index = offset // (batch_size * dim1 * dim2)
    offset_in_channel = offset % (batch_size * dim1 * dim2)

    # Compute the batch, height, width indices
    batch_idx = offset_in_channel // (dim1 * dim2)
    height_idx = (offset_in_channel % (dim1 * dim2)) // dim2
    width_idx = offset_in_channel % dim2

    # Load the value from input
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)

    # Load mean and variance for this channel
    mean = tl.load(mean_ptr + channel_index, mask=mask, other=0.0)
    var = tl.load(var_ptr + channel_index, mask=mask, other=0.0)

    # Normalize
    x_normalized = (x - mean) / tl.sqrt(var + eps)

    # Load gamma and beta
    gamma = tl.load(gamma_ptr + channel_index, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + channel_index, mask=mask, other=0.0)

    # Apply gamma and beta
    y = x_normalized * gamma + beta

    # Store the result
    tl.store(y_ptr + offset, y, mask=mask)


def triton_batchnorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, mean: torch.Tensor, var: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda and mean.is_cuda and var.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()

    # Prepare output tensor
    y = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    num_channels = x.size(1)
    batch_size = x.size(0)
    dim1 = x.size(2)
    dim2 = x.size(3)

    # Determine the number of blocks needed
    grid = lambda meta: ((num_channels,))

    # Launch the Triton kernel
    batchnorm_kernel[grid](x, y, gamma, beta, mean, var, num_channels, batch_size, dim1, dim2, 1e-5, 128)
    return y


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean and variance
        # For simplicity, we use the standard PyTorch method to compute mean and variance
        # In a real implementation, this would be replaced with a Triton kernel
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

        # Update running mean and variance (optional)
        # self.running_mean = 0.9 * self.running_mean + 0.1 * mean
        # self.running_var = 0.9 * self.running_var + 0.1 * var

        # Use running mean and variance for inference
        mean = self.running_mean
        var = self.running_var

        # Normalize using Triton kernel
        y = triton_batchnorm(x, self.gamma, self.beta, mean, var)

        return y