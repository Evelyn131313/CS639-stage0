import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M, K, L,
    BLOCK_M: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each thread block handles a tile of the result matrix
    pid = tl.program_id(0)
    # Compute the batch, row, and column indices
    i = pid // (M * L)
    remaining = pid % (M * L)
    m = remaining // L
    l = remaining % L

    # Compute the sum over k of A[i][m][k] * B[k][l]
    sum_val = 0.0
    for k in range(K):
        a_val = tl.load(A_ptr + i * M * K + m * K + k, mask=k < K, other=0.0)
        b_val = tl.load(B_ptr + k * L + l, mask=k < K, other=0.0)
        sum_val += a_val * b_val
    tl.store(C_ptr + i * M * L + m * L + l, sum_val)

def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((A.size(0), A.size(1), B.size(1)), device=A.device, dtype=A.dtype)
    N, M, K = A.size()
    L = B.size(1)
    grid = (N * M * L, )
    matmul_kernel[grid](A, B, C, N, M, K, L, BLOCK_M=128, BLOCK_L=128, BLOCK_K=128)
    return C

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)