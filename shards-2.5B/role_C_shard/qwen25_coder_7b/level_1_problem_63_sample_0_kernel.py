import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to weight tensor
    y_ptr,  # Pointer to output tensor
    x_shape,  # Shape of the input tensor (batch_size, in_channels, height, width)
    w_shape,  # Shape of the weight tensor (out_channels, in_channels, kernel_size, kernel_size)
    y_shape,  # Shape of the output tensor (batch_size, out_channels, height_out, width_out)
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    dilation,  # Spacing between kernel elements
    groups,  # Number of blocked connections from input channels to output channels
    BLOCK_SIZE: tl.constexpr,
):
    # Unpack shapes
    batch_size, in_channels, height, width = x_shape
    out_channels, _, kernel_size, _ = w_shape
    _, _, height_out, width_out = y_shape

    # Get program ID and offsets
    pid = tl.program_id(axis=0)
    x_coords = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    y_coords = tl.arange(0, BLOCK_SIZE)

    # Ensure we don't go out of bounds
    x_coords = x_coords[:, None] * stride + padding
    y_coords = y_coords[None, :] * stride + padding

    # Load input and weight values
    x = tl.load(x_ptr + x_coords[:, None, None] * width + y_coords[None, :, None], mask=(x_coords[:, None, None] < height) & (y_coords[None, :, None] < width), other=0.0)
    w = tl.load(w_ptr + (tl.arange(0, out_channels)[:, None, None] * in_channels + tl.arange(0, in_channels)[None, :, None]) * kernel_size + tl.arange(0, kernel_size)[None, None, :], mask=(tl.arange(0, out_channels)[:, None, None] < out_channels) & (tl.arange(0, in_channels)[None, :, None] < in_channels) & (tl.arange(0, kernel_size)[None, None, :] < kernel_size), other=0.0)

    # Perform the convolution
    y = tl.zeros((out_channels,), dtype=tl.float32)
    for c in range(in_channels):
        y += x[:, c, :, :] @ w[:, c, :, :]

    # Store the result
    tl.store(y_ptr + pid * BLOCK_SIZE, y)


def triton_conv2d(x: torch.Tensor, w: torch.Tensor, stride: int, padding: int, dilation: int, groups: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    # Prepare output tensor
    in_channels, height, width = x.shape[1], x.shape[2], x.shape[3]
    out_channels, _, kernel_size, _ = w.shape
    height_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    width_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out = torch.empty((x.shape[0], out_channels, height_out, width_out), device=x.device, dtype=x.dtype)

    # Number of elements in the tensors
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](x, w, out, (x.shape[0], in_channels, height, width), (out_channels, in_channels, kernel_size, kernel_size), (x.shape[0], out_channels, height_out, width_out), stride, padding, dilation, groups, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and square kernel using Triton kernels.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        w = self.conv2d.weight
        if self.conv2d.bias is not None:
            b = self.conv2d.bias
            return triton_conv2d(x, w, self.conv2d.stride[0], self.conv2d.padding[0], self.conv2d.dilation[0], self.conv2d.groups) + b
        else:
            return triton_conv2d(x, w, self.conv2d.stride[0], self.conv2d.padding[0], self.conv2d.dilation[0], self.conv2d.groups)