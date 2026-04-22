import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scalar_mult_kernel(
    A_ptr,  # Pointer to input matrix
    s,      # Scalar value
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    # Multiply by scalar
    out = A * s
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_scalar_mult(A: torch.Tensor, s: float):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert A.is_cuda, "Tensor must be on CUDA."
    A = A.contiguous()
    out = torch.empty_like(A)

    # Number of elements in the tensor
    n_elements = A.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    scalar_mult_kernel[grid](A, s, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return triton_scalar_mult(A, s)