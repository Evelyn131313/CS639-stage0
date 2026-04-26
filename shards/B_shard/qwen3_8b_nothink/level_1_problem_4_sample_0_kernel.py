import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to matrix A (M, K)
    B_ptr,  # Pointer to vector B (K, 1)
    C_ptr,  # Pointer to output vector C (M, 1)
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program computes a block of the output vector
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    # Compute the range of rows in A and B that this block will process
    row_offsets = tl.arange(0, BLOCK_SIZE)
    col_offsets = tl.arange(0, K)

    # Load matrix A (M, K)
    A = tl.load(A_ptr + row_offsets[:, None] * K + col_offsets[None, :], mask=row_offsets < M, other=0.0)
    # Load vector B (K, 1)
    B = tl.load(B_ptr + col_offsets, mask=col_offsets < K, other=0.0)

    # Compute the dot product for each row in the block
    C = tl.dot(A, B)
    # Store the result in the output vector
    tl.store(C_ptr + row_offsets, C, mask=row_offsets < M)


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
    C = torch.empty((A.size(0), 1), dtype=A.dtype, device=A.device)

    # Number of elements in the tensor
    M = A.size(0)
    K = A.size(1)
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((M + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    matmul_kernel[grid](A, B, C, M, K, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)