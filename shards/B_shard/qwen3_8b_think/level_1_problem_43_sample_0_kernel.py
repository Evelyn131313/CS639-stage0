import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_pool3d_kernel(
    input_ptr, 
    output_ptr, 
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    input_depth: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_depth: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    output_index = pid * BLOCK_SIZE + offset
    mask = output_index < (batch_size * channels * output_depth * output_height * output_width)
    tl.where(mask, None, tl.exit())

    b = output_index // (channels * output_depth * output_height * output_width)
    rem = output_index % (channels * output_depth * output_height * output_width)
    c = rem // (output_depth * output_height * output_width)
    rem = rem % (output_depth * output_height * output_width)
    d = rem // (output_height * output_width)
    rem = rem % (output_height * output_width)
    h = rem // output_width
    w = rem % output_width

    d_start = d * stride - padding
    h_start = h * stride - padding
    w_start = w * stride - padding

    max_val = -float('inf')
    for i in range(kernel_size):
        d_input = d_start + i * dilation
        if d_input < 0 or d_input >= input_depth:
            continue
        for j in range(kernel_size):
            h_input = h_start + j * dilation
            if h_input < 0 or h_input >= input_height:
                continue
            for k in range(kernel_size):
                w_input = w_start + k * dilation
                if w_input < 0 or w_input >= input_width:
                    continue
                input_idx = (b * channels + c) * input_depth * input_height * input_width + d_input * input_height * input_width + h_input * input_width + w_input
                val = tl.load(input_ptr + input_idx, mask=..., other=-float('inf'))
                if val > max_val:
                    max_val = val

    output_idx = (b * channels + c) * output_depth * output_height * output_width + d * output_height * output_width + h * output_width + w
    tl.store(output_ptr + output_idx, max_val, mask=...)

def triton_max_pool3d(input: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    input_depth = input.shape[2]
    input_height = input.shape[3]
    input_width = input.shape[4]

    output_depth = ((input_depth + 2 * padding - dilation * (kernel_size - 1)) // stride) + 1
    output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1)) // stride) + 1
    output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1)) // stride) + 1

    output = torch.empty_like(input)
    n_elements = output.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    max_pool3d_kernel[grid](input, output, input.shape[0], input.shape[1], input_depth, input_height, input_width, output_depth, output_height, output_width, kernel_size, stride, padding, dilation, BLOCK_SIZE=BLOCK_SIZE)
    return output

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max_pool3d(x, self.kernel_size, self.stride, self.padding, self.dilation)