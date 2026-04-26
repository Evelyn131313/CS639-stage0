import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A,  # pointer to matrix A
    B,  # pointer to matrix B
    C,  # pointer to matrix C
    M,  # number of rows of A and C
    N,  # number of columns of B and C
    K,  # number of columns of A and rows of B
    stride_am,  # stride of A
    stride_ak,  # stride of A
    stride_bm,  # stride of B
    stride_bn,  # stride of B
    stride_cm,  # stride of C
    stride_cn,  # stride of C
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(axis=0)
    # Compute the block size
    block_m = BLOCK_SIZE
    block_n = BLOCK_SIZE
    # Compute the block's starting row and column
    block_start_m = pid // block_n * block_m
    block_start_n = pid % block_n * block_n
    # Compute the block's row and column offsets
    offs_m = block_start_m + tl.arange(0, block_m)
    offs_n = block_start_n + tl.arange(0, block_n)
    # Compute the block's column offsets for A and B
    offs_k = tl.arange(0, K)
    # Initialize accumulator
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    # Load A and B
    for k in range(0, K, BLOCK_SIZE):
        a = tl.load(A + (offs_m[:, None] * stride_am + k * stride_ak + offs_k[None, :]), mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B + (k * stride_bm + offs_k[None, :] * stride_bn + offs_n[:, None]), mask=(offs_k[None, :] < K) & (offs_n[:, None] < N), other=0.0)
        acc += tl.dot(a, b)
    # Store the result
    tl.store(C + (offs_m[:, None] * stride_cm + offs_n[None, :]), acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


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
    C = torch.empty((A.shape[0], B.shape[1]), dtype=A.dtype, device=A.device)

    # Determine the block size
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    num_blocks_m = (A.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_n = (B.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks_m * num_blocks_n,)

    # Launch the Triton kernel
    matmul_kernel[grid](
        A, B, C,
        A.shape[0], B.shape[1], A.shape[1],
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)