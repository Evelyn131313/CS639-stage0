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
        return self.triton_conv3d(x)

    @torch.no_grad()
    def triton_conv3d(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input tensor must be on CUDA."
        batch_size, in_channels, height, width, depth = x.shape
        out_channels = self.out_channels
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        dilation = self.dilation

        # Compute output dimensions
        output_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        output_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        output_depth = depth

        # Prepare kernel and bias
        kernel = torch.randn(out_channels, in_channels // self.groups, kernel_size, kernel_size, 1, device=x.device, dtype=x.dtype)
        if self.bias:
            bias = torch.randn(out_channels, device=x.device, dtype=x.dtype)
        else:
            bias = None

        # Prepare output
        output = torch.empty((batch_size, out_channels, output_h, output_w, output_depth), device=x.device, dtype=x.dtype)

        # Launch Triton kernel
        self.conv3d_kernel[output.shape](x, kernel, output, batch_size, in_channels, out_channels, kernel_size, kernel_size, 1, stride, stride, 1, padding, padding, 0, dilation, dilation, 0, 128)

        if self.bias:
            output += bias.view(1, out_channels, 1, 1, 1)

        return output

    @triton.jit
    def conv3d_kernel(
        input_ptr, 
        kernel_ptr, 
        output_ptr, 
        batch_size, 
        in_channels, 
        out_channels, 
        kernel_h, 
        kernel_w, 
        kernel_d, 
        stride_h, 
        stride_w, 
        stride_d, 
        padding_h, 
        padding_w, 
        padding_d, 
        dilation_h, 
        dilation_w, 
        dilation_d, 
        BLOCK_SIZE: tl.constexpr
    ):
        # Compute grid dimensions
        grid_h = (output_h + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid_w = (output_w + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid_d = (output_depth + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (grid_h, grid_w, grid_d)

        # Compute the index in the output
        pid = tl.program_id(0)
        h = pid % grid_h
        pid = pid // grid_h
        w = pid % grid_w
        pid = pid // grid_w
        d = pid % grid_d

        # Compute the input offset
        offset_input = h * stride_h + kernel_h // 2
        offset_input = w * stride_w + kernel_w // 2
        offset_input = d * stride_d + kernel_d // 2

        # Compute the input indices
        input_indices = tl.arange(0, BLOCK_SIZE)
        input_indices = input_indices + offset_input
        input_indices = input_indices * in_channels + tl.arange(0, in_channels)
        input_indices = input_indices * batch_size + tl.arange(0, batch_size)

        # Load input values
        input_values = tl.load(input_ptr + input_indices, mask=input_indices < x.numel(), other=0.0)

        # Compute kernel indices
        kernel_indices = tl.arange(0, kernel_h)
        kernel_indices = kernel_indices * kernel_w + tl.arange(0, kernel_w)
        kernel_indices = kernel_indices * kernel_d + tl.arange(0, kernel_d)
        kernel_indices = kernel_indices * out_channels + tl.arange(0, out_channels)
        kernel_indices = kernel_indices * in_channels + tl.arange(0, in_channels)
        kernel_indices = kernel_indices * batch_size + tl.arange(0, batch_size)

        # Load kernel values
        kernel_values = tl.load(kernel_ptr + kernel_indices, mask=kernel_indices < kernel.numel(), other=0.0)

        # Compute output values
        output_values = tl.dot(input_values, kernel_values)

        # Store output values
        output_indices = h * stride_h + kernel_h // 2
        output_indices = w * stride_w + kernel_w // 2
        output_indices = d * stride_d + kernel_d // 2
        output_indices = output_indices * out_channels + tl.arange(0, out_channels)
        output_indices = output_indices * batch_size + tl.arange(0, batch_size)

        tl.store(output_ptr + output_indices, output_values)