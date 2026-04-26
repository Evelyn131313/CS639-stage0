import torch
import torch.nn as nn
import triton
import triton.language as tl

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
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.triton_conv(x)

    def triton_conv(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input tensor must be on CUDA."
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias if self.bias is not None else None

        batch, in_c, h, w = x.shape
        out_c, in_c_per_group, kh, kw = weight.shape
        out_h = (h + 2 * self.padding - self.dilation * (kh - 1) - 1) // self.stride + 1
        out_w = (w + 2 * self.padding - self.dilation * (kw - 1) - 1) // self.stride + 1

        out = torch.empty((batch, out_c, out_h, out_w), dtype=x.dtype, device=x.device)

        def triton_kernel(x_ptr, w_ptr, out_ptr, batch, in_c, out_c, kh, kw, stride, padding, dilation, groups, BLOCK_SIZE):
            pid = tl.program_id(0)
            block_h = pid // out_w
            block_w = pid % out_w
            pos_h = block_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            pos_w = block_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = (pos_h < out_h) & (pos_w < out_w)

            x_offsets = tl.arange(0, BLOCK_SIZE)
            w_offsets = tl.arange(0, BLOCK_SIZE)
            x_offsets = x_offsets[:, None] * (out_c * in_c * h * w) + tl.arange(0, out_c)[:, None] * in_c * h * w + tl.arange(0, in_c)[:, None] * h * w + pos_h[None, :] * w + pos_w[None, :]
            w_offsets = w_offsets[:, None] * (out_c * in_c_per_group * kh * kw) + tl.arange(0, out_c)[:, None] * in_c_per_group * kh * kw + tl.arange(0, in_c_per_group)[:, None] * kh * kw + tl.arange(0, kh)[None, :] * kw + tl.arange(0, kw)[None, :]

            x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
            w = tl.load(w_ptr + w_offsets, mask=mask, other=0.0)
            out = tl.sum(x * w, axis=1)
            if bias is not None:
                out += bias[None, None, None, :]
            tl.store(out_ptr + x_offsets, out, mask=mask)

        BLOCK_SIZE = 128
        grid = (out_h * out_w,)
        triton_kernel[grid](x, weight, out, batch, in_c, out_c, kh, kw, self.stride, self.padding, self.dilation, self.groups, BLOCK_SIZE)
        return out