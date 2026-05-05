import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    in_channels,  # Number of input channels
    kernel_size,  # Size of the convolution kernel
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    out = tl.zeros_like(x)
    
    # Perform the depthwise convolution
    for i in range(kernel_size):
        for j in range(kernel_size):
            out += x[i:i+stride, j:j+stride] * weight[i, j]
    
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_depthwise_conv2d(x: torch.Tensor, weight: torch.Tensor):
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
    in_channels = x.shape[1]
    kernel_size = weight.shape[1]
    stride = 1
    padding = 0
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    depthwise_conv2d_kernel[grid](x, weight, out, n_elements, in_channels, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_channels, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return triton_depthwise_conv2d(x, self.weight)