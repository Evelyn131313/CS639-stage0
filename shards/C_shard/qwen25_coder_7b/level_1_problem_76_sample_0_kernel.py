import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to weight tensor
    y_ptr,  # Pointer to output tensor
    x_shape,  # Shape of the input tensor (batch_size, in_channels, length)
    w_shape,  # Shape of the weight tensor (out_channels, in_channels, kernel_size)
    stride,  # Stride of the convolution
    dilation,  # Dilation of the convolution
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch, in_channel, length = x_shape
    out_channel, _, kernel_size = w_shape
    block_id = tl.program_id(0)
    batch_id = block_id // (length // stride)
    out_channel_id = (block_id % (length // stride)) // (length // (stride * kernel_size))
    in_channel_id = (block_id % (length // stride)) % (length // (stride * kernel_size))
    out_length = length // stride

    offsets = tl.arange(0, BLOCK_SIZE)
    x_offsets = (batch_id * in_channel + in_channel_id) * length + offsets
    w_offsets = out_channel_id * in_channel * kernel_size + in_channel_id * kernel_size + offsets
    y_offsets = (batch_id * out_channel + out_channel_id) * out_length + offsets

    x_values = tl.load(x_ptr + x_offsets, mask=offsets < length, other=0.0)
    w_values = tl.load(w_ptr + w_offsets, mask=offsets < kernel_size, other=0.0)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(kernel_size):
        acc += x_values * w_values[k]
    tl.store(y_ptr + y_offsets, acc, mask=offsets < out_length)


def triton_conv1d(x: torch.Tensor, w: torch.Tensor, stride: int, dilation: int):
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
    out_channels, in_channels, kernel_size = w.shape
    batch_size, _, length = x.shape
    out_length = length // stride
    out = torch.empty((batch_size, out_channels, out_length), dtype=x.dtype, device=x.device)

    # Number of elements in the tensor
    n_elements = out.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((out.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv1d_kernel[grid](x, w, out, x.shape, w.shape, stride, dilation, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv1d(x, self.conv1d.weight, self.conv1d.stride[0], self.conv1d.dilation[0])