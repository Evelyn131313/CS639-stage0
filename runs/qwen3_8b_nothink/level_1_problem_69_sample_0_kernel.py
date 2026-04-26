import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height_in: tl.constexpr,
    width_in: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    output_padding_h: tl.constexpr,
    output_padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the output dimensions
    height_out = ((height_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h) // dilation_h + 1
    width_out = ((width_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w) // dilation_w + 1

    # Each thread handles a single output element
    pid = tl.program_id(0)
    output_idx = pid

    # Compute the corresponding input indices
    # Output shape: (batch_size, out_channels, height_out, width_out)
    # Input shape: (batch_size, in_channels, height_in, width_in)
    # Weight shape: (out_channels, in_channels // groups, kernel_h, kernel_w)
    # For simplicity, assume groups = 1
    out_channel = output_idx % out_channels
    output_idx = output_idx // out_channels
    output_h = output_idx % height_out
    output_idx = output_idx // height_out
    output_w = output_idx % width_out

    # Compute input indices
    input_h = (output_h * stride_h) - padding_h
    input_w = (output_w * stride_w) - padding_w

    # Apply dilation
    input_h = input_h + (dilation_h - 1) * (output_h // dilation_h)
    input_w = input_w + (dilation_w - 1) * (output_w // dilation_w)

    # Compute the input and weight indices
    input_h = input_h + (dilation_h - 1) * (output_h // dilation_h)
    input_w = input_w + (dilation_w - 1) * (output_w // dilation_w)

    # For each input channel
    for in_channel in range(in_channels):
        # Compute the weight indices
        weight_h = (output_h * stride_h) - padding_h
        weight_w = (output_w * stride_w) - padding_w
        weight_h = weight_h + (dilation_h - 1) * (output_h // dilation_h)
        weight_w = weight_w + (dilation_w - 1) * (output_w // dilation_w)

        # Load input and weight
        input_val = tl.load(input_ptr + (output_idx * in_channels + in_channel) + (input_h * width_in + input_w), 0.0)
        weight_val = tl.load(weight_ptr + (out_channel * in_channels + in_channel) + (weight_h * kernel_w + weight_w), 0.0)

        # Accumulate the result
        output_val = tl.load(output_ptr + (output_idx * out_channels + out_channel), 0.0)
        output_val += input_val * weight_val
        tl.store(output_ptr + (output_idx * out_channels + out_channel), output_val)


def triton_conv_transpose2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty_like(input)

    # Get parameters
    batch_size = input.size(0)
    in_channels = input.size(1)
    out_channels = weight.size(0)
    height_in = input.size(2)
    width_in = input.size(3)
    kernel_h, kernel_w = weight.size(2), weight.size(3)
    stride_h, stride_w = 1, 1
    padding_h, padding_w = 0, 0
    output_padding_h, output_padding_w = 0, 0
    dilation_h, dilation_w = 1, 1

    # Compute output dimensions
    height_out = ((height_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h) // dilation_h + 1
    width_out = ((width_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w) // dilation_w + 1

    # Compute the number of output elements
    n_elements = batch_size * out_channels * height_out * width_out

    # Determine the number of blocks needed
    BLOCK_SIZE = 128
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    conv_transpose2d_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, height_in, width_in, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, output_padding_h, output_padding_w, dilation_h, dilation_w, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get parameters
        batch_size = x.size(0)
        in_channels = self.in_channels
        out_channels = self.out_channels
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        output_padding_h, output_padding_w = self.output_padding
        dilation_h, dilation_w = self.dilation
        groups = self.groups
        bias = self.bias

        # Initialize weight and bias
        weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_h, kernel_w))
        if bias:
            bias = torch.nn.Parameter(torch.randn(out_channels))

        # Compute output dimensions
        height_in = x.size(2)
        width_in = x.size(3)
        height_out = ((height_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h) // dilation_h + 1
        width_out = ((width_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w) // dilation_w + 1

        # Create output tensor
        output = torch.empty(batch_size, out_channels, height_out, width_out, device=x.device, dtype=x.dtype)

        # Call Triton kernel
        output = triton_conv_transpose2d(x, weight, bias if bias else None)
        return output