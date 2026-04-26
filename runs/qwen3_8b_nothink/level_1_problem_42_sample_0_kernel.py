import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_pool2d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    stride: tl.constexpr,
    kernel_size: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    n_channels: tl.constexpr,
    batch_size: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the position in the output
    pid = tl.program_id(0)
    # Compute the position in the input
    input_pos = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the output position
    output_pos = input_pos // (stride * stride)

    # Compute the input indices
    input_indices = input_pos + (dilation * (kernel_size - 1) // 2) * (stride * stride)
    # Compute the maximum value in the kernel
    max_val = -float('inf')
    for i in range(kernel_size):
        for j in range(kernel_size):
            idx = input_indices + i * stride + j * stride
            val = tl.load(input_ptr + idx, mask=idx < (batch_size * n_channels * height * width), other=-float('inf'))
            if val > max_val:
                max_val = val
    # Store the maximum value in the output
    tl.store(output_ptr + output_pos, max_val)


def triton_max_pool2d(input: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    """
    Custom Triton kernel for Max Pooling 2D.
    """
    assert input.is_cuda, "Input tensor must be on CUDA."
    input = input.contiguous()
    batch_size, n_channels, height, width = input.shape
    output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    output = torch.empty((batch_size, n_channels, output_height, output_width), device=input.device, dtype=input.dtype)

    # Compute the number of elements per block
    num_elements_per_block = (kernel_size * kernel_size) * (BLOCK_SIZE // (stride * stride))
    # Compute the number of blocks
    num_blocks = (batch_size * n_channels * output_height * output_width + num_elements_per_block - 1) // num_elements_per_block

    # Launch the kernel
    max_pool2d_kernel[(num_blocks,)](input, output, stride, kernel_size, padding, dilation, n_channels, batch_size, height, width, BLOCK_SIZE=128)
    return output


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)