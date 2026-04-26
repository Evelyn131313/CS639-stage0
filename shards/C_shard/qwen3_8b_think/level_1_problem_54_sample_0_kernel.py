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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate output dimensions
        batch_size, in_c, d, w, h = x.shape
        out_d = (d + 2 * self.padding - self.dilation * (self.kernel_size - 1) - self.stride) // self.stride + 1
        out_w = (w + 2 * self.padding - self.dilation * (self.kernel_size - 1) - self.stride) // self.stride + 1
        out_h = (h + 2 * self.padding - self.dilation * (self.kernel_size - 1) - self.stride) // self.stride + 1

        # Prepare output tensor
        output = torch.empty((batch_size, self.out_channels, out_d, out_w, out_h), dtype=x.dtype, device=x.device)

        # Launch Triton kernel
        self._launch_triton_kernel(x, output, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, out_d, out_w, out_h)

        return output

    def _launch_triton_kernel(self, x: torch.Tensor, output: torch.Tensor, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int, groups: int, out_d: int, out_w: int, out_h: int):
        # Triton kernel
        @triton.jit
        def conv3d_kernel(
            input_ptr, 
            kernel_ptr, 
            output_ptr, 
            batch_size, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            dilation, 
            groups, 
            out_d, 
            out_w, 
            out_h, 
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            # Compute output indices
            o = pid // (out_d * out_w * out_h)
            rem = pid % (out_d * out_w * out_h)
            d = rem // (out_w * out_h)
            rem = rem % (out_w * out_h)
            w = rem // out_h
            h = rem % out_h

            # Compute input indices
            input_d = d * stride - padding + tl.arange(0, kernel_size) * dilation
            input_w = w * stride - padding + tl.arange(0, kernel_size) * dilation
            input_h = h * stride - padding + tl.arange(0, kernel_size) * dilation

            # Compute kernel indices
            kernel_indices = tl.arange(0, kernel_size)
            kernel_indices = kernel_indices[:, None, None, None]  # (kernel_size, 1, 1, 1)

            # Compute input indices for each kernel position
            input_d = input_d[None, None, None, :] + tl.arange(0, out_d, BLOCK_SIZE)[:, None, None, None] * stride - padding
            input_w = input_w[None, None, None, :] + tl.arange(0, out_w, BLOCK_SIZE)[:, None, None, None] * stride - padding
            input_h = input_h[None, None, None, :] + tl.arange(0, out_h, BLOCK_SIZE)[:, None, None, None] * stride - padding

            # Load input and kernel
            input_vals = tl.load(input_ptr + input_d + input_w + input_h, mask=(input_d < x.shape[2]) & (input_w < x.shape[3]) & (input_h < x.shape[4]), other=0.0)
            kernel_vals = tl.load(kernel_ptr + kernel_indices, mask=(kernel_indices < kernel_size), other=0.0)

            # Compute sum
            sum_vals = tl.sum(input_vals * kernel_vals, axis=0)

            # Store result
            tl.store(output_ptr + o * out_d * out_w * out_h + d * out_w * out_h + w * out_h + h, sum_vals)

        # Launch kernel
        grid = (out_d * out_w * out_h, )
        conv3d_kernel[grid](x, None, output, batch_size, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, out_d, out_w, out_h, BLOCK_SIZE=128)