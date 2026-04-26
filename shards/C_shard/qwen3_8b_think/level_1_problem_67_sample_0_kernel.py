import torch
import torch.nn as nn
import triton
import triton.language as tl

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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        in_channels = x.size(1)
        length = x.size(2)
        out_channels = self.out_channels
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        length_out = (length + 2 * padding - kernel_size) // stride + 1

        out = torch.empty((batch_size, out_channels, length_out), device=x.device, dtype=x.dtype)

        @triton.jit
        def conv1d_kernel(
            x_ptr, kernel_ptr, out_ptr,
            batch_size: tl.constexpr,
            in_channels: tl.constexpr,
            out_channels: tl.constexpr,
            kernel_size: tl.constexpr,
            stride: tl.constexpr,
            padding: tl.constexpr,
            dilation: tl.constexpr,
            length: tl.constexpr,
            length_out: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            thread_idx = tl.program_id(1)
            batch_idx = pid // (out_channels * length_out)
            out_channel_idx = (pid % (out_channels * length_out)) // length_out
            out_pos = pid % length_out

            input_start = out_pos * stride - padding
            padded_length = length + 2 * padding

            in_channel = thread_idx // kernel_size
            kernel_pos = thread_idx % kernel_size
            input_pos = input_start + kernel_pos * dilation

            if input_pos < 0 or input_pos >= padded_length:
                tl.store(out_ptr + batch_idx * out_channels * length_out + out_channel_idx * length_out + out_pos, 0.0)
                return

            x_val = tl.load(x_ptr + batch_idx * in_channels * length + in_channel * length + input_pos, other=0.0)
            kernel_val = tl.load(kernel_ptr + out_channel_idx * in_channels * kernel_size + in_channel * kernel_size + kernel_pos, other=0.0)
            sum_val = x_val * kernel_val

            sum_val = tl.sum(sum_val, axis=0)
            tl.store(out_ptr + batch_idx * out_channels * length_out + out_channel_idx * length_out + out_pos, sum_val)

        grid = (batch_size * out_channels * length_out, 1)
        conv1d_kernel[grid](x, self.weight, out, batch_size, in_channels, out_channels, kernel_size, stride, padding, dilation, length, length_out, BLOCK_SIZE=kernel_size * in_channels)

        if self.bias is not None:
            out += self.bias

        return out