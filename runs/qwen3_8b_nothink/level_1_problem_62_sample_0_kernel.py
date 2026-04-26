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
    kernel_size,  # (height, width)
    stride,  # (height, width)
    padding,  # (height, width)
    dilation,  # (height, width)
    out_channels,  # Number of output channels
    in_channels,  # Number of input channels
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # Compute the 4D index (batch, group, out_channel, out_h, out_w)
    # We use a 5D grid to cover all elements in the output
    # Each thread handles a single output element
    # Each thread block handles a block of input elements
    # We use a 2D block to cover a tile of the input

    # Get the 5D index
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    out_ch_idx = tl.program_id(2)
    out_h_idx = tl.program_id(3)
    out_w_idx = tl.program_id(4)

    # Compute the corresponding input indices
    # Input shape: (batch, in_channels, height, width)
    # Output shape: (batch, out_channels, height_out, width_out)
    # Each output element is computed from a local region of the input

    # Compute the input height and width
    in_h = input_shape[2]
    in_w = input_shape[3]

    # Compute the output height and width
    out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

    # Compute the input starting position
    in_h_start = out_h_idx * stride[0] - padding[0]
    in_w_start = out_w_idx * stride[1] - padding[1]

    # Compute the input region for this output element
    # We use a 2D block to cover the kernel area
    # We loop over the kernel height and width
    # Each thread handles one element of the kernel

    # Compute the input offset for the current batch, group, and output channel
    input_offset = batch_idx * in_channels * in_h * in_w + group_idx * in_channels // GROUPS * in_h * in_w + out_ch_idx * in_h * in_w + out_h_idx * in_w + out_w_idx
    output_offset = batch_idx * out_channels * out_h * out_w + out_ch_idx * out_h * out_w + out_h_idx * out_w + out_w_idx

    # Load the weight for the current output channel
    weight_offset = out_ch_idx * in_channels // GROUPS * kernel_size[0] * kernel_size[1] + group_idx * in_channels // GROUPS * kernel_size[0] * kernel_size[1] + out_h_idx * kernel_size[1] + out_w_idx
    weight = tl.load(weight_ptr + weight_offset, 0.0)

    # Compute the input region for the current output element
    # We loop over the kernel height and width
    # Each thread handles one element of the kernel

    # Initialize the output value
    out_val = 0.0

    # Loop over the kernel height and width
    for kh in range(kernel_size[0]):
        for kw in range(kernel_size[1]):
            # Compute the input position
            in_h_pos = in_h_start + kh * dilation[0]
            in_w_pos = in_w_start + kw * dilation[1]

            # Compute the input offset
            in_offset = input_offset + (in_h_pos * in_w + in_w_pos)
            in_val = tl.load(input_ptr + in_offset, 0.0)

            # Multiply and accumulate
            out_val += in_val * weight

    # Store the result
    tl.store(output_ptr + output_offset, out_val)


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: tuple, padding: tuple, dilation: tuple, groups: int):
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
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels = weight.shape[0]
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation

    out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    output = torch.empty((batch_size, out_channels, out_h, out_w), dtype=input.dtype, device=input.device)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_H = 16
    BLOCK_W = 16
    GROUPS = groups

    # Determine the number of blocks needed
    grid = lambda meta: (batch_size, groups, out_channels, out_h, out_w)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, (batch_size, in_channels, in_h, in_w), (kernel_h, kernel_w), (stride_h, stride_w), (padding_h, padding_w), (dilation_h, dilation_w), out_channels, in_channels, BLOCK_H, BLOCK_W, GROUPS)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
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
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Create a weight tensor with random values
        weight = torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1], device=x.device, dtype=x.dtype)
        # Create a bias tensor if needed
        if self.bias:
            bias = torch.randn(self.out_channels, device=x.device, dtype=x.dtype)
        else:
            bias = None
        # Perform the convolution
        output = triton_conv2d(x, weight, bias, (self.stride, self.stride), (self.padding, self.padding), (self.dilation, self.dilation), self.groups)
        return output