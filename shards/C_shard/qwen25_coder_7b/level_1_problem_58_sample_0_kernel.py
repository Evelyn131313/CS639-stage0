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
    n_in_channels,  # Number of input channels
    kernel_size,  # Kernel size in the form (depth, height, width)
    stride,  # Stride in the form (depth, height, width)
    padding,  # Padding in the form (depth, height, width)
    output_padding,  # Output padding in the form (depth, height, width)
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

    # Initialize output value
    out = tl.zeros_like(x)

    # Load weight values
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            for k in range(kernel_size[2]):
                weight = tl.load(weight_ptr + (i * kernel_size[1] * kernel_size[2] + j * kernel_size[2] + k) * n_in_channels * n_out_channels)
                out += weight * x

    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv_transpose3d(x: torch.Tensor, weight: torch.Tensor, n_out_channels: int, n_in_channels: int, kernel_size: tuple, stride: tuple, padding: tuple, output_padding: tuple):
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
    out = torch.empty(n_out_channels, x.shape[1], x.shape[2] + 2 * padding[0] + output_padding[0], x.shape[3] + 2 * padding[1] + output_padding[1], x.shape[4] + 2 * padding[2] + output_padding[2], device=x.device, dtype=x.dtype)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose3d_kernel[grid](x, weight, out, n_elements, n_out_channels, n_in_channels, kernel_size, stride, padding, output_padding, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth_in, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        weight = self.conv_transpose3d.weight.data
        return triton_conv_transpose3d(x, weight, self.conv_transpose3d.out_channels, self.conv_transpose3d.in_channels, self.conv_transpose3d.kernel_size, self.conv_transpose3d.stride, self.conv_transpose3d.padding, self.conv_transpose3d.output_padding)