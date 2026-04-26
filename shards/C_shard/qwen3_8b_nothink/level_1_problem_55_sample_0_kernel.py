import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch_size, in_channels, height, width)
    weight_shape,  # (out_channels, in_channels // groups, kernel_size, kernel_size)
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    dilation,  # Dilation of the kernel
    groups,  # Number of groups
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    CHANNELS_PER_THREAD: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    # Get the output channel index
    out_channel_idx = tl.program_id(1)
    # Get the output height and width index
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)

    # Compute the input height and width
    in_h = input_shape[2] + 2 * padding
    in_w = input_shape[3] + 2 * padding

    # Compute the output height and width
    out_h = (input_shape[2] + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) // stride + 1
    out_w = (input_shape[3] + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) // stride + 1

    # Compute the input offset for the current output position
    input_h_start = (out_h_idx * stride) - padding
    input_w_start = (out_w_idx * stride) - padding

    # Compute the output offset
    output_offset = batch_idx * input_shape[1] * out_h * out_w + out_channel_idx * out_h * out_w + out_h_idx * out_w + out_w_idx

    # Compute the weight offset
    weight_offset = out_channel_idx * weight_shape[1] * KERNEL_SIZE * KERNEL_SIZE + out_channel_idx % groups * weight_shape[1] * KERNEL_SIZE * KERNEL_SIZE // weight_shape[1] * (out_channel_idx // groups) * weight_shape[1] * KERNEL_SIZE * KERNEL_SIZE // weight_shape[1] * (out_channel_idx % groups)

    # Initialize the output
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the input channels
    for in_channel_idx in range(0, input_shape[1], CHANNELS_PER_THREAD):
        # Compute the input offset for the current input channel
        input_offset = batch_idx * input_shape[1] * in_h * in_w + in_channel_idx * in_h * in_w + input_h_start * in_w + input_w_start

        # Compute the weight offset for the current input channel
        weight_offset_in = weight_offset + in_channel_idx * KERNEL_SIZE * KERNEL_SIZE

        # Load the weight
        weight = tl.load(weight_ptr + weight_offset_in, mask=tl.arange(0, KERNEL_SIZE) < KERNEL_SIZE, other=0.0)

        # Iterate over the kernel
        for k_h in range(KERNEL_SIZE):
            for k_w in range(KERNEL_SIZE):
                # Compute the input offset for the current kernel position
                input_offset_k = input_offset + k_h * in_w + k_w

                # Load the input value
                input_val = tl.load(input_ptr + input_offset_k, other=0.0)

                # Multiply and accumulate
                output += input_val * weight[k_h * KERNEL_SIZE + k_w]

    # Store the output
    tl.store(output_ptr + output_offset, output)


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, stride: int, padding: int, dilation: int, groups: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()

    # Compute output shape
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    in_h = input.shape[2]
    in_w = input.shape[3]
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]
    out_h = (in_h + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Prepare output tensor
    output = torch.empty((batch_size, out_channels, out_h, out_w), dtype=input.dtype, device=input.device)

    # Kernel parameters
    BLOCK_SIZE = 128
    KERNEL_SIZE = kernel_size
    CHANNELS_PER_THREAD = 16

    # Determine the number of blocks needed
    grid = (batch_size, out_channels, out_h, out_w)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, input.shape, weight.shape, stride, padding, dilation, groups, BLOCK_SIZE, KERNEL_SIZE, CHANNELS_PER_THREAD)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create weight tensor
        weight = torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size, device=x.device, dtype=x.dtype)
        # Perform convolution using Triton kernel
        output = triton_conv2d(x, weight, self.stride, self.padding, self.dilation, self.groups)
        # Add bias if needed
        if self.bias:
            output = output + torch.nn.Parameter(torch.randn(self.out_channels, device=x.device, dtype=x.dtype))
        return output