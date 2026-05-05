import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to matrix A
    B_ptr,  # Pointer to matrix B
    C_ptr,  # Pointer to output matrix C
    N: tl.constexpr,  # Size of the square matrices
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a submatrix of size BLOCK_SIZE x BLOCK_SIZE
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Initialize the value of C[row, col]
    C_value = 0.0
    
    # Loop over the common dimension
    for k in range(0, N, BLOCK_SIZE):
        # Load a submatrix of A and B
        A_sub = tl.load(A_ptr + row * N * BLOCK_SIZE + k * BLOCK_SIZE, mask=(row < N and k < N), other=0.0)
        B_sub = tl.load(B_ptr + (k + tl.arange(0, BLOCK_SIZE)) * N + col * BLOCK_SIZE, mask=(k < N and col < N), other=0.0)
        
        # Compute the dot product of A_sub and B_sub
        C_value += tl.dot(A_sub, B_sub)
    
    # Store the result in C[row, col]
    tl.store(C_ptr + row * N * BLOCK_SIZE + col * BLOCK_SIZE, C_value, mask=(row < N and col < N))


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
    C = torch.empty_like(A)

    # Size of the square matrices
    N = A.shape[0]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], (N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])

    # Launch the Triton kernel
    matmul_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)