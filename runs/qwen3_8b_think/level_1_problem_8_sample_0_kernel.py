import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    A_strides, B_strides, C_strides,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N
    c = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load A block
        a_offsets = row_start + tl.arange(0, BLOCK_M) * K + (k + tl.arange(0, BLOCK_K))
        a = tl.load(A_ptr + a_offsets, mask=(row_start + tl.arange(0, BLOCK_M) < M) & (k + tl.arange(0, BLOCK_K) < K), other=0.0)
        # Load B block
        b_offsets = (k + tl.arange(0, BLOCK_K)) * N + tl.arange(0, BLOCK_N)
        b = tl.load(B_ptr + b_offsets, mask=(k + tl.arange(0, BLOCK_K) < K) & (tl.arange(0, BLOCK_N) < N), other=0.0)
        # Compute dot product
        c += tl.dot(a, b)

    # Store result
    c_offsets = row_start * N + col_start + tl.arange(0, BLOCK_M) * N + tl.arange(0, BLOCK_N)
    tl.store(C_ptr + c_offsets, c, mask=(row_start + tl.arange(0, BLOCK_M) < M) & (col_start + tl.arange(0, BLOCK_N) < N))


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype)
    M, K = A.shape
    K, N = B.shape

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    num_row_blocks = (M + BLOCK_M - 1) // BLOCK_M
    num_col_blocks = (N + BLOCK_N - 1) // BLOCK_N

    grid = (num_row_blocks, num_col_blocks)

    matmul_kernel[grid](
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        A.stride(0), B.stride(0), C.stride(0),
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)