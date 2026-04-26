import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    input_ptr,  # Pointer to input tensor (batch, in_channels, length)
    weight_ptr,  # Pointer to weight tensor (out_channels, in_channels // groups, kernel_size)
    output_ptr,  # Pointer to output tensor (batch, out_channels, length_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Compute the block index (batch, out_channel, output_length)
    pid_batch = tl.program_id(0)
    pid_out = tl.program_id(1)
    pid_out_len = tl.program_id(2)

    # Compute the output length
    input_length = (input_ptr.shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    output_length = input_length

    # Compute the input and output indices
    input_start = pid_batch * in_channels * input_length
    input_start += pid_out * in_channels // groups * input_length
    input_start += pid_out_len * input_length
    input_start += tl.arange(0, BLOCK_SIZE)

    # Compute the weight indices
    weight_start = pid_out * in_channels // groups * kernel_size
    weight_start += tl.arange(0, kernel_size)

    # Compute the output index
    output_start = pid_batch * out_channels * output_length
    output_start += pid_out * output_length
    output_start += pid_out_len

    # Load weights
    weights = tl.load(weight_ptr + weight_start, mask=weight_start < out_channels * in_channels // groups * kernel_size, other=0.0)

    # Initialize output
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Compute the convolution
    for k in range(kernel_size):
        # Compute the input offset
        input_offset = input_start + k * dilation
        input_offset += tl.arange(0, BLOCK_SIZE)

        # Load input
        input_vals = tl.load(input_ptr + input_offset, mask=input_offset < in_channels * input_length, other=0.0)

        # Multiply and accumulate
        output += tl.dot(input_vals, weights[k], axis=0)

    # Store output
    tl.store(output_ptr + output_start, output, mask=output_start < out_channels * output_length)


def triton_conv1d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, dilation: int, groups: int):
    """
    Triton implementation of 1D convolution.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Compute output shape
    input_length = input.size(2)
    output_length = (input_length + 2 * padding - dilation * (weight.size(2) - 1) - 1) // stride + 1

    # Prepare output tensor
    output = torch.empty((input.size(0), weight.size(0), output_length), dtype=input.dtype, device=input.device)

    # Parameters
    batch_size = input.size(0)
    in_channels = input.size(1)
    out_channels = weight.size(0)
    kernel_size = weight.size(2)
    BLOCK_SIZE = 128
    num_warps = 4
    num_stages = 4

    # Determine grid dimensions
    grid = (batch_size, out_channels, output_length)

    # Launch kernel
    conv1d_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, BLOCK_SIZE, num_warps, num_stages)

    # Add bias if present
    if bias is not None:
        output += bias.view(1, out_channels, 1)

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
        # Create weight and bias tensors
        weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size))
        if self.bias:
            bias = torch.nn.Parameter(torch.randn(self.out_channels))
        else:
            bias = None

        # Perform convolution with Triton kernel
        return triton_conv1d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)