import torch
import torch.nn as nn
import triton
import triton.language as tl

def triton_conv3d(x: torch.Tensor, kernel: torch.Tensor, stride: int, padding: int, dilation: int):
    assert x.is_cuda and kernel.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    kernel = kernel.contiguous()
    
    batch_size, in_channels, width, height, depth = x.shape
    out_channels, in_channels_k, kW, kH, kD = kernel.shape
    
    # Calculate output dimensions
    out_width = (width + 2 * padding - dilation * (kW - 1) - 1) // stride + 1
    out_height = (height + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    out_depth = (depth + 2 * padding - dilation * (kD - 1) - 1) // stride + 1
    
    output = torch.empty((batch_size, out_channels, out_width, out_height, out_depth), dtype=x.dtype, device=x.device)
    
    # Kernel parameters
    BLOCK_SIZE = 128
    
    @triton.jit
    def conv3d_kernel(
        input_ptr, kernel_ptr, output_ptr,
        batch_size, in_channels, out_channels, kW, kH, kD,
        stride, padding, dilation,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        # Compute output indices
        b = pid // (out_channels * out_width * out_height * out_depth)
        oc = (pid // (out_width * out_height * out_depth)) % out_channels
        w = (pid // (out_height * out_depth)) % out_width
        h = (pid // out_depth) % out_height
        d = pid % out_depth
        
        # Compute input indices
        input_w = w * stride - padding + tl.arange(0, kW)
        input_h = h * stride - padding + tl.arange(0, kH)
        input_d = d * stride - padding + tl.arange(0, kD)
        
        # Check if input indices are within bounds
        mask = (input_w >= 0) & (input_w < width) & (input_h >= 0) & (input_h < height) & (input_d >= 0) & (input_d < depth)
        
        # Load input and kernel values
        input_vals = tl.load(input_ptr + b * in_channels * width * height * depth + 
                            tl.arange(0, kW) * width * height * depth + 
                            tl.arange(0, kH) * height * depth + 
                            tl.arange(0, kD) * depth + 
                            tl.arange(0, in_channels) * width * height * depth, mask=mask, other=0.0)
        kernel_vals = tl.load(kernel_ptr + oc * in_channels_k * kW * kH * kD + 
                              tl.arange(0, kW) * kH * kD + 
                              tl.arange(0, kH) * kD + 
                              tl.arange(0, kD) * in_channels_k + 
                              tl.arange(0, in_channels_k), mask=mask, other=0.0)
        
        # Compute the sum
        output_val = tl.sum(input_vals * kernel_vals)
        
        # Store the result
        tl.store(output_ptr + b * out_channels * out_width * out_height * out_depth + 
                 oc * out_width * out_height * out_depth + 
                 w * out_height * out_depth + 
                 h * out_depth + 
                 d, output_val)
    
    # Launch the kernel
    grid = (batch_size * out_channels * out_width * out_height * out_depth,)
    conv3d_kernel[grid](x, kernel, output, batch_size, in_channels, out_channels, kW, kH, kD, stride, padding, dilation, BLOCK_SIZE=BLOCK_SIZE)
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = triton_conv3d(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output += self.bias
        return output