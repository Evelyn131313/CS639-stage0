import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A,  # pointer to first matrix
    B,  # pointer to second matrix
    C,  # pointer to output matrix
    N,  # size of matrix
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a 16x16 block of the matrix
    pid = tl.program_id(0)
    num_block = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_start = pid * BLOCK_SIZE
    block_row = block_start // BLOCK_SIZE
    block_col = block_start % BLOCK_SIZE

    # Compute the block offset in the matrix
    row_offsets = block_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = block_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load A and B blocks
    A_block = tl.load(A + row_offsets[:, None] * N + col_offsets[None, :], mask=(row_offsets < N)[:, None] & (col_offsets < N)[None, :], other=0.0)
    B_block = tl.load(B + col_offsets[:, None] * N + row_offsets[None, :], mask=(col_offsets < N)[:, None] & (row_offsets < N)[None, :], other=0.0)

    # Compute the dot product
    C_block = tl.dot(A_block, B_block)

    # Write the result to the output matrix
    tl.store(C + row_offsets[:, None] * N + col_offsets[None, :], C_block, mask=(row_offsets < N)[:, None] & (col_offsets < N)[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Custom Triton kernel for matrix multiplication optimized for upper triangular matrices.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices must have the same shape."
    N = A.shape[0]
    out = torch.empty_like(A)

    # Determine block size
    BLOCK_SIZE = 16  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = ( (N + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    # Launch the Triton kernel
    matmul_kernel[grid](A, B, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A * B) for upper triangular matrices using custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices using a custom Triton kernel.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        C = triton_matmul(A, B)
        return torch.triu(C)