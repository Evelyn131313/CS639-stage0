import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch_size, in_channels, depth, width, height)
    kernel_size,  # (kernel_depth, kernel_width, kernel_height)
    stride,  # (stride_depth, stride_width, stride_height)
    padding,  # (padding_depth, padding_width, padding_height)
    dilation,  # (dilation_depth, dilation_width, dilation_height)
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    OUT_CHANNELS: tl.constexpr,
    IN_CHANNELS: tl.constexpr,
    DEPTH: tl.constexpr,
    WIDTH: tl.constexpr,
    HEIGHT: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    # Get the output channel index
    out_ch_idx = tl.program_id(1)
    # Get the output position (depth, width, height)
    out_depth = tl.program_id(2)
    out_width = tl.program_id(3)
    out_height = tl.program_id(4)

    # Compute the output dimensions
    out_depth_dim = (DEPTH + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    out_width_dim = (WIDTH + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    out_height_dim = (HEIGHT + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) // stride[2] + 1

    # Check if the current output position is within bounds
    if out_depth >= out_depth_dim or out_width >= out_width_dim or out_height >= out_height_dim:
        return

    # Compute the input depth, width, height for this output position
    in_depth = out_depth * stride[0] - padding[0] + dilation[0] * (kernel_size[0] - 1)
    in_width = out_width * stride[1] - padding[1] + dilation[1] * (kernel_size[1] - 1)
    in_height = out_height * stride[2] - padding[2] + dilation[2] * (kernel_size[2] - 1)

    # Compute the input channel offset
    in_ch_offset = (out_ch_idx % GROUPS) * (IN_CHANNELS // GROUPS)

    # Compute the output channel offset
    out_ch_offset = out_ch_idx * (IN_CHANNELS // GROUPS)

    # Initialize the output value
    out_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the kernel
    for k_depth in range(kernel_size[0]):
        for k_width in range(kernel_size[1]):
            for k_height in range(kernel_size[2]):
                # Compute the input position
                in_depth_pos = in_depth + k_depth * dilation[0]
                in_width_pos = in_width + k_width * dilation[1]
                in_height_pos = in_height + k_height * dilation[2]

                # Compute the input offset
                in_offset = (batch_idx * IN_CHANNELS + in_ch_offset) * DEPTH * WIDTH * HEIGHT + in_depth_pos * WIDTH * HEIGHT + in_width_pos * HEIGHT + in_height_pos

                # Load input value
                input_val = tl.load(input_ptr + in_offset, mask=tl.arange(0, BLOCK_SIZE) < IN_CHANNELS // GROUPS, other=0.0)

                # Compute the weight offset
                weight_offset = (out_ch_offset + (k_depth * kernel_size[1] + k_width) * kernel_size[2] + k_height) * IN_CHANNELS // GROUPS + in_ch_offset

                # Load weight value
                weight_val = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE, other=0.0)

                # Multiply and accumulate
                out_val += input_val * weight_val

    # Store the output value
    out_offset = (batch_idx * OUT_CHANNELS + out_ch_idx) * out_depth_dim * out_width_dim * out_height_dim + out_depth * out_width_dim * out_height_dim + out_width * out_height_dim + out_height
    tl.store(output_ptr + out_offset, out_val, mask=tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE)


def triton_conv3d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: tuple, padding: tuple, dilation: tuple):
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
    batch_size, in_channels, depth, width, height = input.shape
    kernel_size = (3, 3, 3)
    out_channels = weight.shape[0]
    kernel_depth, kernel_width, kernel_height = kernel_size
    out_depth = (depth + 2 * padding[0] - dilation[0] * (kernel_depth - 1) - 1) // stride[0] + 1
    out_width = (width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
    out_height = (height + 2 * padding[2] - dilation[2] * (kernel_height - 1) - 1) // stride[2] + 1

    # Prepare output tensor
    output = torch.empty((batch_size, out_channels, out_depth, out_width, out_height), dtype=input.dtype, device=input.device)

    # Kernel parameters
    BLOCK_SIZE = 128
    GROUPS = 1
    IN_CHANNELS = in_channels
    OUT_CHANNELS = out_channels
    DEPTH = depth
    WIDTH = width
    HEIGHT = height

    # Determine the number of blocks needed
    grid = lambda meta: (batch_size, out_channels, out_depth, out_width, out_height)

    # Launch the Triton kernel
    conv3d_kernel[grid](input, weight, output, (batch_size, in_channels, depth, width, height), kernel_size, stride, padding, dilation, BLOCK_SIZE, GROUPS, OUT_CHANNELS, IN_CHANNELS, DEPTH, WIDTH, HEIGHT)
    return output


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with square input and square kernel using custom Triton kernels.

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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride, stride)
        self.padding = (padding, padding, padding)
        self.dilation = (dilation, dilation, dilation)
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        # Perform convolution using Triton kernel
        output = triton_conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)
        return output