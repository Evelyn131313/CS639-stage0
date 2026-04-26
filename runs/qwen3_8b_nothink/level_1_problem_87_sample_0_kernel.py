import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 2D grid of threads (block_id_x, block_id_y)
    pid = tl.program_id(0)
    # Each thread handles a block of data of size BLOCK_SIZE
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < width

    # Compute the index in the input tensor
    input_idx = tl.arange(0, BLOCK_SIZE)
    # Compute the index in the weight tensor
    weight_idx = tl.arange(0, BLOCK_SIZE)

    # Compute the index in the output tensor
    # output_idx = (batch_idx, out_channel_idx, height_idx, input_idx)

    # Iterate over all batch elements
    for batch_idx in range(batch_size):
        # Iterate over all output channels
        for out_channel_idx in range(out_channels):
            # Iterate over all input channels
            for in_channel_idx in range(in_channels):
                # Compute the offset in the input tensor
                input_offset = (batch_idx, in_channel_idx, tl.arange(0, height), input_idx)
                # Compute the offset in the weight tensor
                weight_offset = (in_channel_idx, out_channel_idx, weight_idx)
                # Compute the offset in the output tensor
                output_offset = (batch_idx, out_channel_idx, tl.arange(0, height), input_idx)

                # Load input values
                input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
                # Load weight values
                weight_val = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
                # Compute the product
                output_val = input_val * weight_val
                # Accumulate the result into the output
                tl.atomic_add(output_ptr + output_offset, output_val, mask=mask)

    # Return the output
    return output_ptr


def triton_conv1d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Prepare output tensor
    output = torch.empty_like(input)

    # Number of elements in the tensor
    n_elements = input.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv1d_kernel[grid](input, weight, output, input.size(0), input.size(1), input.size(2), input.size(3), input.size(2), BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation using a custom Triton kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, 1, 1).cuda())
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_channels).cuda())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using the custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Apply the custom Triton convolution kernel
        output = triton_conv1d(x, self.weight, self.bias)
        return output