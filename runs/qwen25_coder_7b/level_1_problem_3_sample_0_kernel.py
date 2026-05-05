import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to first input tensor (batch_size, m, k)
    B_ptr,  # Pointer to second input tensor (batch_size, k, n)
    C_ptr,  # Pointer to output tensor (batch_size, m, n)
    batch_size: tl.constexpr,  # Batch size
    m: tl.constexpr,  # Number of rows in A
    k: tl.constexpr,  # Number of columns in A and rows in B
    n: tl.constexpr,  # Number of columns in B
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_row = tl.program_id(0) * BLOCK_SIZE
    block_col = tl.program_id(1) * BLOCK_SIZE

    # Create a range of offsets [0..BLOCK_SIZE-1]
    row_offsets = block_row + tl.arange(0, BLOCK_SIZE)
    col_offsets = block_col + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we don't go out of bounds
    row_mask = row_offsets < m
    col_mask = col_offsets < n

    # Initialize output value to zero
    C = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Compute the dot product
    for i in range(k):
        A_row = tl.load(A_ptr + (row_offsets[:, None] * k + i), mask=row_mask, other=0.0)
        B_col = tl.load(B_ptr + (i * n + col_offsets), mask=col_mask, other=0.0)
        C += A_row * B_col

    # Store the result
    tl.store(C_ptr + (row_offsets[:, None] * n + col_offsets), C, mask=tl.bitwise_and(row_mask[:, None], col_mask))


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
    C = torch.empty_like(A[:, :, :B.shape[2]])

    # Get tensor shapes
    batch_size, m, k = A.shape
    k, n = B.shape

    BLOCK_SIZE = 32  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch the Triton kernel
    matmul_kernel[grid](A, B, C, batch_size, m, k, n, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Instead of "return torch.bmm(A, B)", call our Triton-based matrix multiplication
        return triton_matmul(A, B)