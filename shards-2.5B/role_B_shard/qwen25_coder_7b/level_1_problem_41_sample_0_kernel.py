import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    max_indices_ptr,  # Pointer to max indices tensor (if return_indices=True)
    n_elements,  # Total number of elements in input tensor
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Calculate the start and end indices for the max pooling
    start = tl.floordiv(offsets, stride) * stride - padding + dilation * tl.arange(0, kernel_size)
    end = start + kernel_size

    # Initialize max values and indices
    max_values = tl.full_like(x, float('-inf'))
    max_indices = tl.full_like(offsets, -1, dtype=tl.int32)

    # Perform the max pooling
    for i in range(kernel_size):
        valid_indices = (start + i >= 0) & (start + i < n_elements) & mask
        valid_values = x[valid_indices]
        valid_indices = valid_indices & (valid_values > max_values)
        max_values[valid_indices] = valid_values[valid_indices]
        max_indices[valid_indices] = start[valid_indices] + i

    # Store the result
    tl.store(out_ptr + offsets, max_values, mask=mask)

    # Store the max indices if return_indices is True
    if max_indices_ptr is not None:
        tl.store(max_indices_ptr + offsets, max_indices, mask=mask)


def triton_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    output_sequence_length = ((x.size(2) + padding * 2 - dilation * (kernel_size - 1) - 1) // stride) + 1
    out = torch.empty((x.size(0), x.size(1), output_sequence_length), dtype=x.dtype, device=x.device)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    maxpool1d_kernel[grid](x, out, None, n_elements, kernel_size, stride, padding, dilation, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)