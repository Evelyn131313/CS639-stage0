import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    input_h,  # Input height
    input_w,  # Input width
    output_h,  # Output height
    output_w,  # Output width
    kernel_h,  # Kernel height
    kernel_w,  # Kernel width
    stride_h,  # Stride height
    stride_w,  # Stride width
    padding_h,  # Padding height
    padding_w,  # Padding width
    output_padding_h,  # Output padding height
    output_padding_w,  # Output padding width
    groups,  # Number of groups
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a single output element
    pid = tl.program_id(0)
    # Compute the output position (o_h, o_w)
    o_h = pid // output_w
    o_w = pid % output_w

    # Compute the input position range for this output
    i_h_start = (o_h * stride_h) - (kernel_h - 1) + padding_h
    i_w_start = (o_w * stride_w) - (kernel_w - 1) + padding_w

    # Compute the input position range
    i_h_end = i_h_start + kernel_h
    i_w_end = i_w_start + kernel_w

    # Initialize the output value
    out = tl.zeros((out_channels,), dtype=tl.float32)

    # Iterate over the input channels
    for c in range(in_channels // groups):
        # Load the weight for this channel
        weight = tl.load(weight_ptr + c * out_channels * kernel_h * kernel_w + tl.arange(0, kernel_h * kernel_w), None)

        # Iterate over the input positions
        for i_h in range(i_h_start, i_h_end):
            for i_w in range(i_w_start, i_w_end):
                # Compute the input index
                input_idx = (i_h * input_w + i_w) * in_channels + c
                input_val = tl.load(input_ptr + input_idx, None)

                # Compute the kernel index
                k_h = i_h - (o_h * stride_h)
                k_w = i_w - (o_w * stride_w)

                # Compute the weight index
                weight_idx = k_h * kernel_w + k_w
                weight_val = weight[weight_idx]

                # Accumulate the result
                out += input_val * weight_val

    # Store the result
    output_idx = (o_h * output_w + o_w) * out_channels + tl.arange(0, out_channels)
    tl.store(output_ptr + output_idx, out, None)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, output_padding: int, groups: int):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    output = torch.empty((x.size(0), weight.size(0), x.size(2) + output_padding, x.size(3) + output_padding), dtype=x.dtype, device=x.device)

    # Compute output dimensions
    output_h = (x.size(2) - 1) * stride + weight.size(2) - 2 * padding + output_padding
    output_w = (x.size(3) - 1) * stride + weight.size(3) - 2 * padding + output_padding

    # Determine the number of blocks needed
    grid = (output_h * output_w,)

    # Launch the Triton kernel
    conv_transpose2d_kernel[grid](x, weight, output, x.size(0), x.size(1), weight.size(0), x.size(2), x.size(3), output_h, output_w, weight.size(2), weight.size(3), stride, padding, output_padding, groups, BLOCK_SIZE=128)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the custom Triton-based transposed convolution
        if self.bias is not None:
            return triton_conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups)
        else:
            return triton_conv_transpose2d(x, self.weight, None, self.stride, self.padding, self.output_padding, self.groups)