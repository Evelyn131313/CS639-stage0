import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_tril_kernel(
    A_ptr,  # Pointer to matrix A
    B_ptr,  # Pointer to matrix B
    C_ptr,  # Pointer to output matrix C
    N: tl.constexpr,  # Size of square matrix
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(axis=0)
    # Compute the block offset
    block_start = pid * BLOCK_SIZE
    # Compute the block row and column indices
    row = block_start // N
    col = block_start % N
    # Compute the offset in the matrix
    offsets = row * N + col + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < N * N
    # Load A and B
    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    B = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    # Compute the dot product
    C = tl.dot(A, B)
    # Store the result
    tl.store(C_ptr + offsets, C, mask=mask)


def triton_matmul_tril(A: torch.Tensor, B: torch.Tensor):
    """
    Custom Triton kernel for matrix multiplication of lower triangular matrices.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    # Create output tensor
    C = torch.empty_like(A)
    # Number of elements in the tensor
    N = A.size(0)
    # Block size (tunable)
    BLOCK_SIZE = 128
    # Determine the number of blocks needed
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel
    grid = (num_blocks,)
    matmul_tril_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul_tril(A, B)