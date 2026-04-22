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
    for in_channel_idx in range(weight_shape[1]):
        # Compute the input offset for the current channel
        input_offset = batch_idx * input_shape[1] * in_h * in_w + in_channel_idx * in_h * in_w + input_h_start * in_w + input_w_start

        # Iterate over the kernel
        for kh in range(KERNEL_SIZE):
            for kw in range(KERNEL_SIZE):
                # Compute the weight offset for the current kernel position
                weight_offset_k = weight_offset + in_channel_idx * KERNEL_SIZE * KERNEL_SIZE + kh * KERNEL_SIZE + kw

                # Load the weight
                weight = tl.load(weight_ptr + weight_offset_k, dtype=tl.float32)

                # Compute the input offset for the current kernel position
                input_offset_k = input_offset + (kh * dilation + input_h_start) * in_w + (kw * dilation + input_w_start)

                # Load the input
                input_val = tl.load(input_ptr + input_offset_k, dtype=tl.float32)

                # Multiply and accumulate
                output += input_val * weight

    # Store the result
    tl.store(output_ptr + output_offset, output)


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, dilation: int, groups: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Prepare output tensor
    output = torch.empty_like(input)

    # Get input and weight shapes
    input_shape = input.shape
    weight_shape = weight.shape

    # Compute the number of output elements
    batch_size = input_shape[0]
    in_channels = input_shape[1]
    out_channels = weight_shape[0]
    height = input_shape[2]
    width = input_shape[3]

    # Compute output shape
    out_h = (height + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) // stride + 1
    out_w = (width + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) // stride + 1

    # Determine the number of blocks needed
    grid = (batch_size, out_channels, out_h, out_w)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, input_shape, weight_shape, stride, padding, dilation, groups, 128, 3, 1)
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

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Apply padding
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)

        # Perform convolution
        output = triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output