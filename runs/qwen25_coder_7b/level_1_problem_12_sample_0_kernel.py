import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_diag_kernel(
    A_ptr,  # Pointer to diagonal elements
    B_ptr,  # Pointer to second matrix
    C_ptr,  # Pointer to output matrix
    N: tl.constexpr,  # Number of rows in A and B, and columns in B
    M: tl.constexpr,  # Number of columns in B
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_row = tl.program_id(0)
    block_col = tl.program_id(1)
    # Create a range of offsets [0..BLOCK_SIZE-1]
    row_offsets = block_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = block_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    row_mask = row_offsets < N
    col_mask = col_offsets < M
    # Load diagonal elements and B values
    diag = tl.load(A_ptr + row_offsets, mask=row_mask, other=0.0)
    B = tl.load(B_ptr + row_offsets[:, None] * M + col_offsets, mask=tl.bitwise_and(row_mask[:, None], col_mask), other=0.0)
    # Perform the matrix multiplication
    C = tl.dot(diag[:, None], B)
    # Store the result
    tl.store(C_ptr + row_offsets[:, None] * M + col_offsets, C, mask=tl.bitwise_and(row_mask[:, None], col_mask))


def triton_matmul_diag(A: torch.Tensor, B: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    # Prepare output tensor
    C = torch.empty_like(B)

    # Number of rows in A and B, and columns in B
    N, M = A.numel(), B.shape[1]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], (M + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])

    # Launch the Triton kernel
    matmul_diag_kernel[grid](A, B, C, N, M, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return triton_matmul_diag(A, B)