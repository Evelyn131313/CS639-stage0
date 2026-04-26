import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # [batch, in_channels, depth, width, height]
    kernel_size,  # Size of the kernel
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    dilation,  # Dilation of the kernel
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # Compute the batch, input channel, depth, width, height indices
    batch_idx = tl.program_id(0)
    in_channel_idx = tl.program_id(1)
    depth_idx = tl.program_id(2)
    width_idx = tl.program_id(3)
    height_idx = tl.program_id(4)

    # Compute the output channel index
    out_channel_idx = in_channel_idx % GROUPS * (out_channels // GROUPS) + (in_channel_idx // GROUPS) * (in_channels // GROUPS)

    # Compute the input offset for the current batch, input channel, depth, width, height
    input_offset = (
        batch_idx * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] +
        in_channel_idx * input_shape[2] * input_shape[3] * input_shape[4] +
        depth_idx * input_shape[3] * input_shape[4] +
        width_idx * input_shape[4] +
        height_idx
    )

    # Compute the weight offset for the current output channel, input channel, kernel depth, kernel width, kernel height
    weight_offset = (
        out_channel_idx * in_channels * kernel_size * kernel_size * kernel_size +
        in_channel_idx * kernel_size * kernel_size * kernel_size +
        tl.arange(0, kernel_size) * kernel_size * kernel_size +
        tl.arange(0, kernel_size) * kernel_size +
        tl.arange(0, kernel_size)
    )

    # Compute the output offset for the current batch, output channel, depth, width, height
    output_offset = (
        batch_idx * out_channels * input_shape[2] * input_shape[3] * input_shape[4] +
        out_channel_idx * input_shape[2] * input_shape[3] * input_shape[4] +
        depth_idx * input_shape[3] * input_shape[4] +
        width_idx * input_shape[4] +
        height_idx
    )

    # Initialize the output value
    out_val = 0.0

    # Iterate over the kernel dimensions
    for d in range(kernel_size):
        for w in range(kernel_size):
            for h in range(kernel_size):
                # Compute the input depth, width, height
                input_depth = depth_idx + d * dilation
                input_width = width_idx + w * dilation
                input_height = height_idx + h * dilation

                # Compute the input offset for the current kernel position
                input_offset_kernel = (
                    batch_idx * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] +
                    in_channel_idx * input_shape[2] * input_shape[3] * input_shape[4] +
                    input_depth * input_shape[3] * input_shape[4] +
                    input_width * input_shape[4] +
                    input_height
                )

                # Load the input value
                input_val = tl.load(input_ptr + input_offset_kernel, 0.0)

                # Load the weight value
                weight_val = tl.load(weight_ptr + weight_offset, 0.0)

                # Multiply and accumulate
                out_val += input_val * weight_val

    # Store the result
    tl.store(output_ptr + output_offset, out_val)


def triton_conv3d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, dilation: int):
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

    # Compute the output shape
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    depth = input.shape[2]
    width = input.shape[3]
    height = input.shape[4]
    out_channels = weight.shape[0]

    # Compute the output depth, width, height
    out_depth = (depth + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Prepare output tensor
    output = torch.empty((batch_size, out_channels, out_depth, out_width, out_height), dtype=input.dtype, device=input.device)

    # Number of elements in the tensor
    n_elements = batch_size * in_channels * depth * width * height

    # Determine the number of blocks needed
    grid = lambda meta: (batch_size, in_channels, depth, width, height)

    # Launch the Triton kernel
    conv3d_kernel[grid](input, weight, output, (batch_size, in_channels, depth, width, height), kernel_size, stride, padding, dilation, BLOCK_SIZE=128, GROUPS=1)
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
        weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size, self.kernel_size))
        if self.bias:
            bias = torch.nn.Parameter(torch.randn(self.out_channels))
        else:
            bias = None

        # Apply the Triton-based 3D convolution
        output = triton_conv3d(x, weight, bias, self.stride, self.padding, self.dilation)
        return output