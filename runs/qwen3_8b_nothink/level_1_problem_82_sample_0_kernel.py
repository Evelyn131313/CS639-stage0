import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    stride: tl.constexpr,
    kernel_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    IN_CHANNELS: tl.constexpr,
    OUT_CHANNELS: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
):
    # Compute the 2D grid of blocks
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)
    pid3 = tl.program_id(2)

    # Compute the block coordinates
    block_h = pid
    block_w = pid2
    block_g = pid3

    # Compute the starting position in the input
    offset_h = block_h * BLOCK_SIZE
    offset_w = block_w * BLOCK_SIZE
    offset_g = block_g * GROUP_SIZE

    # Compute the starting position in the output
    out_offset_h = block_h * BLOCK_SIZE
    out_offset_w = block_w * BLOCK_SIZE

    # Compute the input and output strides
    input_strides = (IN_CHANNELS * HEIGHT * WIDTH, HEIGHT * WIDTH, WIDTH, 1)
    output_strides = (OUT_CHANNELS * HEIGHT * WIDTH, HEIGHT * WIDTH, WIDTH, 1)

    # Compute the input and output offsets
    input_offset = offset_g + (offset_h * WIDTH + offset_w) * IN_CHANNELS
    output_offset = offset_g + (out_offset_h * WIDTH + out_offset_w) * OUT_CHANNELS

    # Compute the range of elements to process in the block
    range_h = tl.arange(0, BLOCK_SIZE)
    range_w = tl.arange(0, BLOCK_SIZE)

    # Compute the mask for valid indices
    mask_h = range_h < HEIGHT
    mask_w = range_w < WIDTH

    # Initialize the output block
    out_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Iterate over the input block and apply the convolution
    for i in range_h:
        for j in range_w:
            input_offset_i = input_offset + i * WIDTH + j
            input_vals = tl.load(input_ptr + input_offset_i, mask=mask_h & mask_w, other=0.0)
            for k in range(BLOCK_SIZE):
                out_block[i, k] += input_vals[k] * tl.load(input_ptr + input_offset_i + k * WIDTH, mask=mask_h & mask_w, other=0.0)

    # Store the output block
    tl.store(output_ptr + output_offset, out_block, mask=mask_h & mask_w)


def triton_depthwise_conv(input: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    Custom Triton kernel for depthwise 2D convolution.
    """
    assert input.is_cuda, "Input tensor must be on CUDA."
    input = input.contiguous()
    output = torch.empty_like(input)

    # Calculate output dimensions
    out_h = (input.shape[2] + 2 * padding - kernel_size) // stride + 1
    out_w = (input.shape[3] + 2 * padding - kernel_size) // stride + 1

    # Prepare input and output strides
    input_strides = (input.shape[1] * input.shape[2] * input.shape[3], input.shape[2] * input.shape[3], input.shape[3], 1)
    output_strides = (input.shape[1] * input.shape[2] * input.shape[3], input.shape[2] * input.shape[3], input.shape[3], 1)

    # Determine block size and grid size
    BLOCK_SIZE = 16
    GROUP_SIZE = input.shape[1]
    num_blocks_h = (out_h + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_w = (out_w + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_g = (GROUP_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    grid = (num_blocks_h, num_blocks_w, num_blocks_g)
    depthwise_conv_kernel[grid](
        input, output, stride, kernel_size, BLOCK_SIZE, GROUP_SIZE, input.shape[1], input.shape[1], out_h, out_w
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv(x, self.kernel_size, self.stride, self.padding)