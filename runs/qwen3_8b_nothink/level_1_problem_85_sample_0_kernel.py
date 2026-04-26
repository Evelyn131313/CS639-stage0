import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output dimensions
        batch_size, in_channels, height, width = x.shape
        out_channels = self.out_channels
        kernel_h, kernel_w = self.kernel_size_h, self.kernel_size_w
        stride_h, stride_w = self.stride_h, self.stride_w
        padding_h, padding_w = self.padding_h, self.padding_w
        dilation_h, dilation_w = self.dilation_h, self.dilation_w
        groups = self.groups

        # Compute output dimensions
        out_h = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        out_w = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

        # Pad input if needed
        x = torch.nn.functional.pad(x, (padding_w, padding_w, padding_h, padding_h), mode='constant', value=0)

        # Initialize output
        output = torch.zeros(batch_size, out_channels, out_h, out_w, device=x.device)

        # Launch Triton kernel
        self.depthwise_conv_kernel[triton.next_power_of_two(out_h * out_w)](x, self.weight, self.bias, output, batch_size, in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups)

        return output


@triton.jit
def depthwise_conv_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    output_ptr,  # Pointer to output tensor
    batch_size,  # Batch size
    in_channels,  # Input channels
    out_channels,  # Output channels
    kernel_h,  # Kernel height
    kernel_w,  # Kernel width
    stride_h,  # Stride height
    stride_w,  # Stride width
    padding_h,  # Padding height
    padding_w,  # Padding width
    dilation_h,  # Dilation height
    dilation_w,  # Dilation width
    groups,  # Number of groups
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the index in the output
    pid = tl.program_id(0)
    # Compute the position in the output
    out_h = pid // (out_channels * out_w)
    out_w = (pid // out_channels) % out_w
    out_c = pid % out_channels

    # Compute the starting position in the input
    in_h = out_h * stride_h - padding_h
    in_w = out_w * stride_w - padding_w

    # Compute the number of elements in the kernel
    kernel_elements = kernel_h * kernel_w
    # Compute the number of elements per group
    elements_per_group = (in_channels // groups) * kernel_elements

    # Compute the index in the input
    in_c = out_c * (in_channels // groups)
    in_c_group = in_c % (in_channels // groups)
    in_c_group = in_c_group * kernel_elements

    # Compute the index in the weight
    weight_c = in_c_group // kernel_elements
    weight_c = weight_c * kernel_elements
    weight_c = weight_c + in_c_group % kernel_elements

    # Compute the index in the output
    output_idx = out_c * out_w * out_h + out_h * out_w + out_w
    output_idx = output_idx * batch_size

    # Compute the index in the input
    input_idx = in_c * (height * width) + in_h * width + in_w
    input_idx = input_idx * batch_size

    # Load input
    x = tl.load(x_ptr + input_idx, 0.0)
    # Load weight
    weight = tl.load(weight_ptr + weight_c, 0.0)
    # Compute the product
    out = x * weight
    # Add bias if present
    if bias_ptr is not None:
        out += tl.load(bias_ptr + out_c, 0.0)
    # Store output
    tl.store(output_ptr + output_idx, out, 0.0)


def triton_depthwise_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, out_channels: int, kernel_h: int, kernel_w: int, stride_h: int, stride_w: int, padding_h: int, padding_w: int, dilation_h: int, dilation_w: int, groups: int):
    batch_size, in_channels, height, width = x.shape
    out_channels = out_channels
    kernel_h, kernel_w = kernel_h, kernel_w
    stride_h, stride_w = stride_h, stride_w
    padding_h, padding_w = padding_h, padding_w
    dilation_h, dilation_w = dilation_h, dilation_w
    groups = groups

    # Compute output dimensions
    out_h = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    # Initialize output
    output = torch.zeros(batch_size, out_channels, out_h, out_w, device=x.device)

    # Launch Triton kernel
    depthwise_conv_kernel[triton.next_power_of_two(out_h * out_w)](x, weight, bias, output, batch_size, in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups)

    return output