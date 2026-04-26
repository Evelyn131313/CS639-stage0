import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread block handles a tile of the output matrix
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    # Compute the i and j indices for this thread block
    i = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Loop over the k dimension
    for k in range(N):
        # Load A[i][k] and B[k][j]
        a = tl.load(A_ptr + i * N + k, mask=k < N, other=0.0)
        b = tl.load(B_ptr + k * N + j, mask=k < N, other=0.0)
        # Multiply and accumulate
        acc += a * b

    # Store the result into C
    tl.store(C_ptr + i * N + j, acc, mask=(i < N) & (j < N))

def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    # Prepare output tensor
    C = torch.empty_like(A)

    # Number of elements in the tensor
    N = A.size(0)
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ( (N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], (N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )

    # Launch the Triton kernel
    matmul_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)