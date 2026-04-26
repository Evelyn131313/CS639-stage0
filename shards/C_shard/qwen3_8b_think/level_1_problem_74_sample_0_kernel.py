import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr,  # Pointer to input tensor (batch, in_channels, input_length)
    kernel_ptr,  # Pointer to kernel (out_channels, in_channels, kernel_size)
    bias_ptr,  # Pointer to bias (out_channels)
    output_ptr,  # Pointer to output tensor (batch, out_channels, output_length)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    input_length: tl.constexpr,
    output_length: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread processes one output element
    pid = tl.program_id(0)
    # Calculate the output element index
    out_idx = pid
    # Calculate the input indices for this output element
    # For each kernel position k
    for k in range(kernel_size):
        # Compute the input position
        i_input = out_idx * stride - (k - (kernel_size - 1)) * dilation
        if i_input < 0 or i_input >= input_length:
            continue
        # Load input value
        input_val = tl.load(input_ptr + out_idx * in_channels * input_length + i_input, other=0.0)
        # Load kernel value
        kernel_val = tl.load(kernel_ptr + out_idx * in_channels * kernel_size + k, other=0.0)
        # Multiply and accumulate
        output_val += input_val * kernel_val
    # Add bias if applicable
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_idx, other=0.0)
        output_val += bias_val
    # Store the result
    tl.store(output_ptr + out_idx, output_val)


def triton_conv_transpose(x: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, dilation: int):
    batch_size, in_channels, input_length = x.size()
    out_channels, _, kernel_size = kernel.size()
    # Compute output length
    output_length = (input_length - 1) * stride + kernel_size - 2 * padding + dilation * (kernel_size - 1) + 1
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=x.device, dtype=x.dtype)
    # Launch Triton kernel
    # Calculate grid size
    grid = (output_length,)
    # Launch kernel
    conv_transpose_kernel[grid](x, kernel, bias, output, batch_size, in_channels, out_channels, kernel_size, stride, padding, dilation, input_length, output_length, BLOCK_SIZE=128)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, input_length = x.size()
        out_channels = self.out_channels
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        # Compute output length
        output_length = (input_length - 1) * stride + kernel_size - 2 * padding + dilation * (kernel_size - 1) + 1
        # Prepare output tensor
        output = torch.empty(batch_size, out_channels, output_length, device=x.device, dtype=x.dtype)
        # Launch Triton kernel
        output = triton_conv_transpose(x, self.kernel, self.bias, stride, padding, dilation)
        return output