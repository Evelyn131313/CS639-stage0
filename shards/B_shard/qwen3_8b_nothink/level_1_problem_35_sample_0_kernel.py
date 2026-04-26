import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    input_ptr,  # Pointer to input tensor
    gamma_ptr,  # Pointer to gamma (scale) tensor
    beta_ptr,   # Pointer to beta (shift) tensor
    output_ptr, # Pointer to output tensor
    batch_size: tl.constexpr,
    num_groups: tl.constexpr,
    num_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the group index, channel index, and spatial index
    pid = tl.program_id(0)
    group_id = pid // (height * width)
    group_idx = pid % (height * width)

    # Compute the channel index within the group
    channel_idx = group_id % num_channels
    group_idx = group_id // num_channels

    # Compute the offset in the input tensor
    input_offset = group_idx * num_channels * height * width + channel_idx * height * width + group_idx * width
    input_offset += tl.arange(0, BLOCK_SIZE)

    # Compute the spatial index
    spatial_idx = input_offset % width
    channel_offset = (input_offset // width) % height
    group_offset = input_offset // (height * width)

    # Load input values
    input_vals = tl.load(input_ptr + input_offset, mask=input_offset < batch_size * num_channels * height * width, other=0.0)

    # Compute mean and variance for the group
    mean = tl.sum(input_vals) / BLOCK_SIZE
    var = tl.sum((input_vals - mean) * (input_vals - mean)) / BLOCK_SIZE

    # Normalize
    normalized = (input_vals - mean) * tl.rsqrt(var + 1e-5)

    # Apply gamma and beta
    gamma = tl.load(gamma_ptr + group_idx, other=1.0)
    beta = tl.load(beta_ptr + group_idx, other=0.0)
    output = gamma * normalized + beta

    # Store the result
    tl.store(output_ptr + input_offset, output, mask=input_offset < batch_size * num_channels * height * width)


def triton_group_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # Prepare output tensor
    output = torch.empty_like(x)

    # Parameters
    batch_size = x.size(0)
    num_channels = x.size(1)
    height = x.size(2)
    width = x.size(3)
    num_groups = gamma.size(0)
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((batch_size * num_channels * height * width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    group_norm_kernel[grid](x, gamma, beta, output, batch_size, num_groups, num_channels, height, width, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs Group Normalization using custom Triton kernels.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer with custom Triton kernel.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.gamma = nn.Parameter(torch.ones(num_groups))
        self.beta = nn.Parameter(torch.zeros(num_groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return triton_group_norm(x, self.gamma, self.beta)