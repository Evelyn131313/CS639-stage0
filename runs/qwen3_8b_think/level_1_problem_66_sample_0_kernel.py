import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        super().__init__()
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
            self.bias = None

    def forward(self, x):
        batch, in_c, d, h, w = x.shape
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding
        dd, dh, dw = self.dilation
        groups = self.groups

        # Compute output dimensions
        out_d = (d + 2 * pd - kd) // sd + 1
        out_h = (h + 2 * ph - kh) // sh + 1
        out_w = (w + 2 * pw - kw) // sw + 1

        output = torch.empty((batch, self.out_channels, out_d, out_h, out_w), device=x.device, dtype=x.dtype)

        # Launch Triton kernel
        triton_conv3d(x, self.weight, output, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
        return output

@triton.jit
def triton_conv3d_kernel(
    x_ptr, w_ptr, out_ptr,
    x_shape: tl.constexpr, w_shape: tl.constexpr,
    kernel_size: tl.constexpr, stride: tl.constexpr, padding: tl.constexpr, dilation: tl.constexpr,
    groups: tl.constexpr, bias_ptr: tl.constexpr
):
    # Extract dimensions
    batch, in_c, d, h, w = x_shape
    out_c, in_c_per_group, kd, kh, kw = w_shape
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    num_groups = groups

    # Compute output dimensions
    out_d = (d + 2 * pd - kd) // sd + 1
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    # Compute thread indices
    pid = tl.program_id(0)
    block_d = pid // (out_h * out_w)
    block_h = (pid // out_w) % out_h
    block_w = pid % out_w

    # Compute offset for output
    offset_d = block_d * sd
    offset_h = block_h * sh
    offset_w = block_w * sw

    # Compute input offsets for each dimension
    input_d_start = offset_d - pd
    input_h_start = offset_h - ph
    input_w_start = offset_w - pw

    # Load weights
    weights = tl.load(w_ptr, shape=(out_c, in_c_per_group, kd, kh, kw), aligned=True)

    # Compute output value
    out_val = tl.zeros((out_c,), dtype=tl.float32)
    for g in range(num_groups):
        in_c_start = g * in_c_per_group
        in_c_end = in_c_start + in_c_per_group
        for d_idx in range(kd):
            for h_idx in range(kh):
                for w_idx in range(kw):
                    input_d = input_d_start + d_idx * dd
                    input_h = input_h_start + h_idx * dh
                    input_w = input_w_start + w_idx * dw
                    x_val = tl.load(x_ptr + input_d + input_h + input_w, dtype=tl.float32)
                    w_val = tl.load(weights[g, d_idx, h_idx, w_idx], dtype=tl.float32)
                    out_val += x_val * w_val

    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr, dtype=tl.float32)
        out_val += bias_val

    # Store result
    out_idx = block_d * out_h * out_w + block_h * out_w + block_w
    tl.store(out_ptr + out_idx, out_val)

def triton_conv3d(x, weight, output, kernel_size, stride, padding, dilation, groups, bias):
    # Prepare kernel call
    x_shape = (x.size(0), x.size(1), x.size(2), x.size(3), x.size(4))
    w_shape = (weight.size(0), weight.size(1), kernel_size[0], kernel_size[1], kernel_size[2])
    grid = (x.size(0), x.size(1), x.size(2), x.size(3), x.size(4))
    triton_conv3d_kernel[grid](x, weight, output, x_shape, w_shape, kernel_size, stride, padding, dilation, groups, bias)