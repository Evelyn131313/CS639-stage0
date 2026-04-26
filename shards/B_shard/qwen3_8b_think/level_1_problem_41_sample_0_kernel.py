import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def maxpool1d_kernel(
    x_ptr,  # Pointer to input tensor (1D)
    out_ptr,  # Pointer to output tensor (1D)
    n_elements,  # Number of elements in the input
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output position
    i = tl.program_id(0)
    # Compute the start index in the input
    start = i * stride - padding
    # Compute the end index in the input
    end = start + (kernel_size - 1) * dilation
    # Check if the window is within bounds
    max_val = -float('inf')
    for j in range(kernel_size):
        idx = start + j * dilation
        if idx < 0 or idx >= n_elements:
            continue
        val = tl.load(x_ptr + idx, mask=idx < n_elements, other=-float('inf'))
        if val > max_val:
            max_val = val
    # Store the result
    tl.store(out_ptr + i, max_val)

def triton_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    input_length = x.size(2)
    output_length = ((input_length + 2 * padding - (kernel_size - 1) * dilation) // stride) + 1
    out = torch.empty((x.size(0), x.size(1), output_length), device=x.device, dtype=x.dtype)
    # Now, for each batch and feature, apply the kernel
    for b in range(x.size(0)):
        for f in range(x.size(1)):
            # Extract the input sub-tensor
            x_sub = x[b, f, :]
            out_sub = out[b, f, :]
            # Launch the kernel
            grid = (output_length,)
            maxpool1d_kernel[grid](x_sub, out_sub, input_length, kernel_size, stride, padding, dilation, BLOCK_SIZE=128)
    return out

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)