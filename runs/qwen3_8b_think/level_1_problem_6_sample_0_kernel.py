import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # Shared memory for A and B blocks
    shared_A = tl.shared((BLOCK_K,))
    shared_B = tl.shared((BLOCK_K,))

    # Load A and B blocks into shared memory
    # For each row in the output block
    for row in range(BLOCK_M):
        # Load A[row_start + row] into shared_A
        a_off = (row_start + row) * K + tl.arange(0, BLOCK_K)
        shared_A[tl.arange(0, BLOCK_K)] = tl.load(A_ptr + a_off, mask=tl.arange(0, BLOCK_K) < K, other=0.0)
        tl.debug_barrier()

        # For each column in the output block
        for col in range(BLOCK_N):
            # Load B[col_start + col] into shared_B
            b_off = tl.arange(0, BLOCK_K) * N + (col_start + col)
            shared_B[tl.arange(0, BLOCK_K)] = tl.load(B_ptr + b_off, mask=tl.arange(0, BLOCK_K) < K, other=0.0)
            tl.debug_barrier()

            # Compute dot product
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(BLOCK_K):
                acc += shared_A[k] * shared_B[k]
            tl.debug_barrier()

            # Store result
            c_off = (row_start + row) * N + (col_start + col)
            tl.store(C_ptr + c_off, acc, mask=tl.arange(0, BLOCK_M) < M and tl.arange(0, BLOCK_N) < N)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    # Prepare output tensor
    C = torch.empty((A.shape[0], B.shape[1]), dtype=A.dtype, device=A.device)

    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    # Calculate grid size
    num_blocks_m = (A.shape[0] + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (B.shape[1] + BLOCK_N - 1) // BLOCK_N
    grid = (num_blocks_m, num_blocks_n)

    # Launch kernel
    matmul_kernel[grid](A, B, C, A.shape[0], B.shape[1], B.shape[0], BLOCK_M, BLOCK_N, BLOCK_K)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)