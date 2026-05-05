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
    kernel_size,  # Size of the convolution kernel
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    dilation,  # Spacing between kernel elements
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
    # Initialize output value
    out = tl.zeros((n_out_channels,), dtype=tl.float32)
    # Perform the convolution
    for c in range(n_out_channels):
        for k in range(n_in_channels):
            for i in range(kernel_size):
                for j in range(kernel_size):
                    out[c] += x[k, i, j] * weight[c, k, i, j]
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, stride: int, padding: int, dilation: int):
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
    n_in_channels = weight.shape[1]
    kernel_size = weight.shape[2]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose2d_kernel[grid](x, weight, out, n_elements, n_out_channels, n_in_channels, kernel_size, stride, padding, dilation, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size).cuda())
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in). 

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv_transpose2d(x, self.weight, self.stride, self.padding, self.dilation)