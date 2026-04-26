import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.triton_max_pool(x)

    def triton_max_pool(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input must be on CUDA."
        batch, channels, height, width = x.shape
        out_h = (height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_w = (width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out = torch.empty(batch, channels, out_h, out_w, dtype=x.dtype, device=x.device)
        self.max_pool_kernel[triton.next_power_of_two(batch * channels * out_h * out_w)](x, out, batch, channels, height, width, self.kernel_size, self.stride, self.padding, self.dilation)
        return out

    @triton.jit
    def max_pool_kernel(
        x_ptr,  # pointer to input
        out_ptr,  # pointer to output
        batch,  # batch size
        channels,  # number of channels
        height,  # input height
        width,  # input width
        kernel_size,  # kernel size
        stride,  # stride
        padding,  # padding
        dilation,  # dilation
        BLOCK_SIZE: tl.constexpr,
    ):
        # Compute output dimensions
        out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        # Compute grid size
        grid_size = triton.next_power_of_two(batch * channels * out_h * out_w)
        # Compute thread indices
        pid = tl.program_id(0)
        # Compute the offset in the output tensor
        out_idx = pid
        # Compute the batch, channel, i, j
        b = out_idx // (channels * out_h * out_w)
        c = (out_idx % (channels * out_h * out_w)) // (out_h * out_w)
        i = (out_idx % (out_h * out_w)) // out_w
        j = out_idx % out_w
        # Compute the start position in input
        start_h = i * stride - padding
        start_w = j * stride - padding
        # Compute the window elements
        max_val = -float('inf')
        for h in range(kernel_size):
            for w in range(kernel_size):
                input_h = start_h + h * dilation
                input_w = start_w + w * dilation
                if input_h < 0 or input_h >= height or input_w < 0 or input_w >= width:
                    continue
                idx = (b * channels + c) * height * width + input_h * width + input_w
                val = tl.load(x_ptr + idx, mask=..., other=-float('inf'))
                if val > max_val:
                    max_val = val
        # Compute the output index
        out_idx = (b * channels + c) * out_h * out_w + i * out_w + j
        tl.store(out_ptr + out_idx, max_val)