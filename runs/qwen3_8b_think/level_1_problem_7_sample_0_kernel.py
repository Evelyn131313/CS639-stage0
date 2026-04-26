import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return self.triton_matmul(A, B)

    def triton_matmul(self, A, B):
        assert A.is_cuda and B.is_cuda
        C = torch.empty((A.size(0), B.size(1)), device=A.device, dtype=A.dtype)
        BLOCK_M = 128
        BLOCK_N = 128
        grid_m = (A.size(0) + BLOCK_M - 1) // BLOCK_M
        grid_n = (B.size(1) + BLOCK_N - 1) // BLOCK_N
        matmul_kernel[grid_m, grid_n](A, B, C, A.size(0), B.size(1), B.size(0), BLOCK_M, BLOCK_N)
        return C


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    i = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    j = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K):
        a = tl.load(A_ptr + i * K + k)
        b = tl.load(B_ptr + k * N + j)
        acc += a * b
    for m in range(BLOCK_M):
        for n in range(BLOCK_N):
            tl.store(C_ptr + i[m] * N + j[n], acc[m][n])