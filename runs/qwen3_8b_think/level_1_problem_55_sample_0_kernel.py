import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr,  # pointer to input tensor
    w_ptr,  # pointer to weights tensor
    y_ptr,  # pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Compute the output position (i, j, out_c)
    i = pid // (out_channels * (out_channels // groups))
    j = (pid // (out_channels // groups)) % (out_channels // groups)
    out_c = pid % (out_channels // groups)
    # Compute the input position (i, j) with padding and dilation
    input_i = i * stride - padding + tl.arange(0, kernel_size) * dilation
    input_j = j * stride - padding + tl.arange(0, kernel_size) * dilation
    # Compute the input channel (in_c)
    in_c = out_c * groups + tl.arange(0, groups)
    # Compute the weight indices
    weight_indices = (out_c, in_c, tl.arange(0, kernel_size), tl.arange(0, kernel_size))
    # Load input and weights
    x = tl.load(x_ptr + input_i[:, None] * in_channels + input_j[None, :] * batch_size + in_c[None, :], mask=(input_i < x.shape[2] and input_j < x.shape[3]), other=0.0)
    w = tl.load(w_ptr + weight_indices, mask=(input_i < w.shape[2] and input_j < w.shape[3]), other=0.0)
    # Compute the output
    y = tl.sum(x * w, axis=(0, 1))
    # Add bias if present
    if bias is not None:
        y += bias[out_c]
    # Store the result
    tl.store(y_ptr + i * out_channels + j + out_c * out_channels, y)

def triton_conv2d(x, w, bias, stride, padding, dilation, groups):
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    y = torch.empty_like(x)
    # Compute grid size
    grid = (x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3],)
    # Launch the Triton kernel
    conv2d_kernel[grid](x, w, y, x.shape[0], x.shape[1], w.shape[0], w.shape[2], stride, padding, dilation, groups, BLOCK_SIZE=128)
    return y

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
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform the convolution using the Triton kernel
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)