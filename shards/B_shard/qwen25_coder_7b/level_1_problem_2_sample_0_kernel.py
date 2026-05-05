import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to matrix A
    B_ptr,  # Pointer to matrix B
    C_ptr,  # Pointer to output matrix C
    M: tl.constexpr,  # Number of rows in A and C
    N: tl.constexpr,  # Number of columns in B and C
    K: tl.constexpr,  # Number of columns in A and rows in B
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
    k_mask = tl.arange(0, K) < K

    # Load input values
    A = tl.load(A_ptr + row_offsets[:, None] * K + k_mask, mask=row_mask[:, None] & k_mask, other=0.0)
    B = tl.load(B_ptr + k_mask[None, :] * N + col_offsets, mask=k_mask[None, :] & col_mask, other=0.0)

    # Initialize output to zero
    C = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Perform the matrix multiplication
    for k in range(0, K, BLOCK_SIZE):
        A_local = A
        B_local = B[:, k:k + BLOCK_SIZE]
        C += A_local[:, None, :] * B_local[None, :, :]

    # Store the result
    tl.store(C_ptr + row_offsets[:, None] * N + col_offsets, C, mask=row_mask[:, None] & col_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
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
    C = torch.empty((A.shape[0], B.shape[1]), dtype=torch.float32, device=A.device)

    # Determine the number of blocks needed
    BLOCK_SIZE = 128  # Tunable parameter for block size
    grid = (A.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE, (B.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    matmul_kernel[grid](A, B, C, A.shape[0], B.shape[1], A.shape[1], BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)