import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_transpose_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    n_out_channels,  # Number of output channels
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
    out = tl.zeros_like(x)
    # Perform the transposed convolution
    for i in range(kernel_size):
        out += x * weight[i]
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv1d_transpose(x: torch.Tensor, weight: torch.Tensor, n_out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int):
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
    length_out = (x.shape[-1] - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    out = torch.empty((x.shape[0], n_out_channels, length_out), device=x.device, dtype=x.dtype)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv1d_transpose_kernel[grid](x, weight, out, n_elements, n_out_channels, kernel_size, stride, padding, dilation, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        weight = self.conv1d_transpose.weight
        bias = self.conv1d_transpose.bias if self.conv1d_transpose.bias is not None else 0.0
        return triton_conv1d_transpose(x, weight, self.conv1d_transpose.out_channels, self.conv1d_transpose.kernel_size, self.conv1d_transpose.stride, self.conv1d_transpose.padding, self.conv1d_transpose.dilation)