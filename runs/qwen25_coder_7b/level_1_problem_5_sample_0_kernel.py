import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_scalar_kernel(
    A_ptr,  # Pointer to input matrix A
    s,  # Scalar value
    C_ptr,  # Pointer to output matrix C
    M: tl.constexpr,  # Number of rows in A and C
    N: tl.constexpr,  # Number of columns in A and C
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    row = tl.program_id(0)
    col = tl.program_id(1)
    # Create a range of offsets [0..BLOCK_SIZE-1]
    row_offsets = row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    row_mask = row_offsets < M
    col_mask = col_offsets < N
    # Load input values
    A = tl.load(A_ptr + row_offsets[:, None] * N + col_offsets[None, :], mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    # Perform the matrix-scalar multiplication
    C = A * s
    # Store the result
    tl.store(C_ptr + row_offsets[:, None] * N + col_offsets[None, :], C, mask=row_mask[:, None] & col_mask[None, :])


def triton_matmul_scalar(A: torch.Tensor, s: float):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert A.is_cuda, "Tensor must be on CUDA."
    A = A.contiguous()

    # Prepare output tensor
    C = torch.empty_like(A)

    # Number of rows and columns in the tensor
    M, N = A.shape
    BLOCK_SIZE = 32  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((M + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], (N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])

    # Launch the Triton kernel
    matmul_scalar_kernel[grid](A, s, C, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using Triton kernel.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return triton_matmul_scalar(A, s)