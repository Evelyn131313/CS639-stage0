import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    n_out_channels,  # Number of output channels
    kernel_size,  # Kernel size (kernel_depth, kernel_width, kernel_height)
    stride,  # Stride (stride_depth, stride_width, stride_height)
    padding,  # Padding (padding_depth, padding_width, padding_height)
    output_padding,  # Output padding (output_padding_depth, output_padding_width, output_padding_height)
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
    # Load weight values
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    # Initialize output to zero
    out = tl.zeros_like(x)
    # Perform the transposed 3D convolution
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            for k in range(kernel_size[2]):
                out += x * weight[i, j, k]
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv_transpose3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    n_out_channels = weight.shape[0]
    kernel_size = weight.shape[1:]
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    output_padding = (0, 0, 0)
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose3d_kernel[grid](x, weight, out, n_elements, n_out_channels, kernel_size, stride, padding, output_padding, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        return triton_conv_transpose3d(x, self.conv_transpose3d.weight, self.conv_transpose3d.bias)