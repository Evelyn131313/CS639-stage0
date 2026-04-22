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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(x)

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, in_channels, height, width = x.shape
        out_channels = self.out_channels
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        groups = self.groups

        # Compute output dimensions
        out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        # Initialize output tensor
        output = torch.empty((batch, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)

        # Launch Triton kernel
        self._triton_conv2d(x, self.weight, output, batch, in_channels, out_channels, height, width, kernel_size, stride, padding, dilation, groups)
        return output

    def _triton_conv2d(self, input, weight, output, batch, in_channels, out_channels, height, width, kernel_size, stride, padding, dilation, groups):
        assert input.is_cuda and weight.is_cuda and output.is_cuda, "Tensors must be on CUDA."
        input = input.contiguous()
        weight = weight.contiguous()
        output = output.contiguous()

        # Calculate output dimensions
        out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        # Define kernel
        @triton.jit
        def conv_kernel(
            input_ptr,  # pointer to input tensor
            weight_ptr,  # pointer to weight tensor
            output_ptr,  # pointer to output tensor
            batch,  # batch size
            in_channels,  # input channels
            out_channels,  # output channels
            height,  # input height
            width,  # input width
            kernel_size,  # kernel size
            stride,  # stride
            padding,  # padding
            dilation,  # dilation
            groups,  # groups
            BLOCK_SIZE: tl.constexpr,
        ):
            # Each program handles a contiguous block of data
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < (out_height * out_width * out_channels)
            # Compute the output index
            out_idx = offsets // (out_channels * out_width)
            out_channel = (offsets // out_width) % out_channels
            out_y = (offsets % out_width) // out_width
            out_x = offsets % out_width
            # Compute the input index
            in_y = out_y * stride - padding
            in_x = out_x * stride - padding
            # Compute the kernel indices
            kernel_y = tl.arange(0, kernel_size)
            kernel_x = tl.arange(0, kernel_size)
            # Compute the input indices for each kernel position
            in_y = in_y + kernel_y * dilation
            in_x = in_x + kernel_x * dilation
            # Load input values
            input_vals = tl.load(input_ptr + (in_y * width + in_x), mask=mask, other=0.0)
            # Load weight values
            weight_vals = tl.load(weight_ptr + (out_channel * in_channels // groups * kernel_size * kernel_size + in_channel * kernel_size * kernel_size + kernel_y * kernel_size + kernel_x), mask=mask, other=0.0)
            # Compute the sum
            sum_val = tl.sum(input_vals * weight_vals)
            # Store the result
            tl.store(output_ptr + offsets, sum_val, mask=mask)

        # Launch kernel
        grid = lambda meta: (out_channels * out_height * out_width,)
        conv_kernel[grid](input, weight, output, batch, in_channels, out_channels, height, width, kernel_size, stride, padding, dilation, groups, BLOCK_SIZE=128)