import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.triton_conv_transpose2d(x)

    @triton.jit
    def conv_transpose2d_kernel(
        input_ptr,  # Pointer to input tensor
        weight_ptr,  # Pointer to weight tensor
        output_ptr,  # Pointer to output tensor
        input_shape,  # Shape of input tensor (batch, in_channels, height, width)
        output_shape,  # Shape of output tensor (batch, out_channels, height_out, width_out)
        kernel_size_h, kernel_size_w,  # Kernel size
        stride_h, stride_w,  # Stride
        padding_h, padding_w,  # Padding
        dilation_h, dilation_w,  # Dilation
        groups,  # Groups
        BLOCK_SIZE: tl.constexpr,
        num_warps: tl.constexpr
    ):
        # Compute output dimensions
        batch, in_c, in_h, in_w = input_shape
        out_c, _, _, _ = output_shape
        out_h = ((in_h - 1) * stride_h + kernel_size_h - 2 * padding_h + dilation_h * (kernel_size_h - 1) + 1)
        out_w = ((in_w - 1) * stride_w + kernel_size_w - 2 * padding_w + dilation_w * (kernel_size_w - 1) + 1)

        # Compute output indices
        pid = tl.program_id(0)
        block_h = pid // out_w
        block_w = pid % out_w

        # Compute input indices
        h_start = block_h * stride_h - padding_h
        w_start = block_w * stride_w - padding_w

        # Compute the number of elements in the block
        block_h_size = min(out_h - block_h * stride_h, BLOCK_SIZE)
        block_w_size = min(out_w - block_w * stride_w, BLOCK_SIZE)

        # Create a grid of indices for the block
        h_offsets = tl.arange(0, BLOCK_SIZE)
        w_offsets = tl.arange(0, BLOCK_SIZE)
        h_offsets = h_offsets + h_start
        w_offsets = w_offsets + w_start

        # Load weights
        weight = tl.load(weight_ptr + tl.arange(0, out_c * in_c // groups) * kernel_size_h * kernel_size_w + tl.arange(0, kernel_size_h) * kernel_size_w + tl.arange(0, kernel_size_w), mask=(tl.arange(0, out_c * in_c // groups) < out_c * in_c // groups) & (tl.arange(0, kernel_size_h) < kernel_size_h) & (tl.arange(0, kernel_size_w) < kernel_size_w), other=0.0)

        # Load input values
        input = tl.load(input_ptr + (h_offsets * in_w + w_offsets) * in_c + tl.arange(0, in_c), mask=(h_offsets < in_h) & (w_offsets < in_w) & (tl.arange(0, in_c) < in_c), other=0.0)

        # Compute output
        output = tl.sum(input * weight, axis=1)

        # Store output
        tl.store(output_ptr + (block_h * out_w + block_w) * out_c + tl.arange(0, out_c), output, mask=(block_h < out_h) & (block_w < out_w) & (tl.arange(0, out_c) < out_c))

    def triton_conv_transpose2d(self, x: torch.Tensor) -> torch.Tensor:
        batch, in_c, in_h, in_w = x.shape
        out_c = self.out_channels
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        groups = self.groups

        # Compute output dimensions
        out_h = ((in_h - 1) * stride_h + kernel_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1)
        out_w = ((in_w - 1) * stride_w + kernel_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1)

        # Prepare output tensor
        output = torch.empty((batch, out_c, out_h, out_w), dtype=x.dtype, device=x.device)

        # Launch the Triton kernel
        grid = (out_h * out_w,)
        conv_transpose2d_kernel[grid](
            x, self.weight, output,
            (batch, in_c, in_h, in_w),
            (batch, out_c, out_h, out_w),
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups,
            BLOCK_SIZE=128,
            num_warps=4
        )

        return output