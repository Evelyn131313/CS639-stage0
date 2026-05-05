import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    n_out_channels,  # Number of output channels
    n_in_channels,  # Number of input channels
    kernel_height,  # Height of the convolution kernel
    kernel_width,  # Width of the convolution kernel
    stride_height,  # Stride height
    stride_width,  # Stride width
    padding_height,  # Padding height
    padding_width,  # Padding width
    BLOCK_SIZE_X: tl.constexpr,  # Block size in X dimension
    BLOCK_SIZE_Y: tl.constexpr,  # Block size in Y dimension
):
    # Get the coordinates of the current thread within the block
    x_coord = tl.program_id(0) * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    y_coord = tl.program_id(1) * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    # Initialize the output value to zero
    out = tl.zeros((BLOCK_SIZE_X, BLOCK_SIZE_Y), dtype=tl.float32)

    # Iterate over the input channels and kernel elements
    for c in range(n_in_channels):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate the input coordinates
                in_x = x_coord + (y_coord * stride_width) + (kh - padding_height)
                in_y = x_coord + (y_coord * stride_height) + (kw - padding_width)

                # Load the input and weight values
                x_val = tl.load(x_ptr + (c * n_elements + in_x * n_elements + in_y), mask=(in_x >= 0) & (in_x < height) & (in_y >= 0) & (in_y < width), other=0.0)
                weight_val = tl.load(weight_ptr + (c * n_out_channels * kernel_height * kernel_width + n_out_channels * kh * kernel_width + n_out_channels * kw), mask=True, other=0.0)

                # Accumulate the dot product
                out += x_val * weight_val

    # Store the result
    out_ptr += (tl.program_id(1) * n_out_channels * n_elements + tl.program_id(0) * n_elements)
    tl.store(out_ptr, out)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, stride: int = 1, padding: int = 0, output_padding: int = 0):
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
    n_batch, n_in_channels, height, width = x.shape
    kernel_height, kernel_width = weight.shape[2:]
    n_out_channels = weight.shape[0]
    out_height = (height - 1) * stride - 2 * padding + kernel_height + output_padding
    out_width = (width - 1) * stride - 2 * padding + kernel_width + output_padding
    out = torch.empty((n_batch, n_out_channels, out_height, out_width), dtype=torch.float32, device=x.device)

    # Number of elements in the tensor
    n_elements = n_batch * n_in_channels * height * width
    BLOCK_SIZE_X = 32  # Tunable parameter for block size in X dimension
    BLOCK_SIZE_Y = 32  # Tunable parameter for block size in Y dimension

    # Determine the number of blocks needed
    grid = lambda meta: ((out_height + meta["BLOCK_SIZE_Y"] - 1) // meta["BLOCK_SIZE_Y"], (out_width + meta["BLOCK_SIZE_X"] - 1) // meta["BLOCK_SIZE_X"])

    # Launch the Triton kernel
    conv_transpose2d_kernel[grid](x, weight, out, n_elements, n_out_channels, n_in_channels, kernel_height, kernel_width, stride, stride, padding, padding, BLOCK_SIZE_X=BLOCK_SIZE_X, BLOCK_SIZE_Y=BLOCK_SIZE_Y)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]).cuda())
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv_transpose2d(x, self.weight, self.stride, self.padding, self.output_padding)