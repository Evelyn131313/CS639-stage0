import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr,  # Pointer to input tensor
    y_ptr,  # Pointer to output tensor
    B: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes a single (b, d2) position
    block_id = tl.program_id(0)
    b = block_id // D2
    d2 = block_id % D2

    # Compute the starting offset for this (b, d2) position in the input tensor
    offset = b * D1 * D2 + d2 * D1
    # Compute the starting offset in the output tensor
    out_offset = b * D2 + d2

    # Initialize max_val and max_idx for this (b, d2) position
    max_val = -float('inf')
    max_idx = -1

    # Process the d1 dimension in blocks of size BLOCK_SIZE
    # Each block processes a range of d1 elements
    # The block index within the d1 dimension
    block_in_d1 = tl.program_id(1)

    # Compute the starting index in d1 for this block
    start_d1 = block_in_d1 * BLOCK_SIZE
    # Compute the end index in d1 for this block
    end_d1 = start_d1 + BLOCK_SIZE

    # Load the values in the block
    for d1 in range(start_d1, end_d1):
        if d1 >= D1:
            break
        x_val = tl.load(x_ptr + offset + d1, mask=d1 < D1, other=-float('inf'))
        if x_val > max_val:
            max_val = x_val
            max_idx = d1

    # Store the result in the output tensor
    tl.store(y_ptr + out_offset, max_idx)


def triton_argmax(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Determine the shape of the output tensor
    if dim == 1:
        B, D1, D2 = x.shape
        output_shape = (B, D2)
    elif dim == 2:
        B, D1, D2 = x.shape
        output_shape = (B, D1)
    elif dim == 0:
        D1, D2 = x.shape
        output_shape = (D1, D2)
    else:
        raise ValueError("Unsupported dimension for argmax")

    # Prepare output tensor
    y = torch.empty(output_shape, dtype=torch.int64, device=x.device)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    # For dim=1: B * D2 blocks
    # For dim=2: B * D1 blocks
    # For dim=0: D2 blocks
    if dim == 1:
        grid = (B * D2, (D1 + BLOCK_SIZE - 1) // BLOCK_SIZE)
    elif dim == 2:
        grid = (B * D1, (D2 + BLOCK_SIZE - 1) // BLOCK_SIZE)
    elif dim == 0:
        grid = (D2, (D1 + BLOCK_SIZE - 1) // BLOCK_SIZE)
    else:
        raise ValueError("Unsupported dimension for argmax")

    # Launch the Triton kernel
    argmax_kernel[grid](x, y, B, D1, D2, BLOCK_SIZE=BLOCK_SIZE)
    return y


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmax(x, self.dim)