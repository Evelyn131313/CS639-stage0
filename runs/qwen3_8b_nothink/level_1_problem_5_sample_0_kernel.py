import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scalar_mult_kernel(
    A_ptr,  # Pointer to input matrix A
    s_ptr,  # Pointer to scalar s
    out_ptr,  # Pointer to output matrix
    M: tl.constexpr,  # Number of rows
    N: tl.constexpr,  # Number of columns
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(axis=0)
    # Compute the row index
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the column indices
    col = tl.arange(0, BLOCK_SIZE)
    # Create mask for valid indices
    mask = (row < M) & (col < N)
    # Load scalar value
    s = tl.load(s_ptr)
    # Load input matrix A
    A = tl.load(A_ptr + row[:, None] * N + col[None, :], mask=mask, other=0.0)
    # Multiply by scalar
    out = A * s
    # Store result
    tl.store(out_ptr + row[:, None] * N + col[None, :], out, mask=mask)


def triton_scalar_mult(A: torch.Tensor, s: float):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert A.is_cuda, "Tensor must be on CUDA."
    A = A.contiguous()
    # Prepare output tensor
    out = torch.empty_like(A)
    # Number of rows and columns
    M = A.shape[0]
    N = A.shape[1]
    # Tunable block size
    BLOCK_SIZE = 128  # Can be adjusted based on GPU memory and performance

    # Determine the number of blocks needed
    grid = lambda meta: ((M + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    scalar_mult_kernel[grid](A, torch.tensor(s).cuda(), out, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        # Use the optimized Triton-based scalar multiplication
        return triton_scalar_mult(A, s)