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
    kernel_size,  # Size of the square kernel
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    dilation,  # Dilation factor
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # Compute the 4D indices
    batch_idx = tl.program_id(0)
    in_channel_idx = tl.program_id(1)
    out_channel_idx = tl.program_id(2)
    height_idx = tl.program_id(3)
    width_idx = tl.program_id(4)

    # Compute the offset in the input tensor
    batch = batch_idx
    in_channel = in_channel_idx % GROUPS
    in_channel_group = in_channel_idx // GROUPS
    out_channel = out_channel_idx
    height = height_idx
    width = width_idx

    # Compute the input and output dimensions
    batch_size, in_channels, input_height, input_width = input_shape
    out_channels = (out_channel_idx + 1)  # out_channel_idx is 0-based
    kernel_h = kernel_size
    kernel_w = kernel_size
    out_h = (input_height + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    out_w = (input_width + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1

    # Compute the output position
    out_h_idx = height_idx
    out_w_idx = width_idx

    # Compute the input position
    input_h = height_idx * stride - padding
    input_w = width_idx * stride - padding

    # Iterate over the output channels
    for out_channel in range(out_channels):
        # Iterate over the input channels
        for in_channel in range(in_channels):
            # Compute the weight index
            weight_in = in_channel
            weight_out = out_channel
            weight_h = 0
            weight_w = 0

            # Compute the input positions for the kernel
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # Compute the input position
                    input_h_k = input_h + kh * dilation
                    input_w_k = input_w + kw * dilation

                    # Check if the input position is within bounds
                    if input_h_k < 0 or input_h_k >= input_height or input_w_k < 0 or input_w_k >= input_width:
                        continue

                    # Compute the input offset
                    input_offset = (
                        batch * in_channels * input_height * input_width
                        + in_channel_group * in_channels // GROUPS * input_height * input_width
                        + in_channel * input_height * input_width
                        + input_h_k * input_width
                        + input_w_k
                    )

                    # Load input value
                    input_val = tl.load(input_ptr + input_offset, mask=0 <= input_offset < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], other=0.0)

                    # Compute the weight offset
                    weight_offset = (
                        in_channel_group * out_channels // GROUPS * in_channels * kernel_h * kernel_w
                        + out_channel * in_channels * kernel_h * kernel_w
                        + in_channel * kernel_h * kernel_w
                        + kh * kernel_w
                        + kw
                    )

                    # Load weight value
                    weight_val = tl.load(weight_ptr + weight_offset, mask=0 <= weight_offset < out_channels * in_channels * kernel_h * kernel_w, other=0.0)

                    # Multiply and accumulate
                    output_val = tl.load(output_ptr + output_offset, mask=0 <= output_offset < batch_size * out_channels * out_h * out_w, other=0.0)
                    output_val += input_val * weight_val
                    tl.store(output_ptr + output_offset, output_val, mask=0 <= output_offset < batch_size * out_channels * out_h * out_w)

    return


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, dilation: int):
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

    # Compute output dimensions
    batch_size, in_channels, input_height, input_width = input.shape
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]
    out_h = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Prepare output tensor
    output = torch.empty((batch_size, out_channels, out_h, out_w), dtype=input.dtype, device=input.device)

    # Compute the grid
    grid = (batch_size, in_channels, out_channels, out_h, out_w)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, (batch_size, in_channels, input_height, input_width), kernel_size, stride, padding, dilation, BLOCK_SIZE=128, GROUPS=1)
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
        # Initialize weight and bias
        weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size))
        if self.bias:
            bias = torch.nn.Parameter(torch.randn(self.out_channels))
        else:
            bias = None

        # Perform convolution using Triton kernel
        output = triton_conv2d(x, weight, bias, self.stride, self.padding, self.dilation)
        return output