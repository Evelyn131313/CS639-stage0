import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread computes a single element of the output
    # We use pid = (block_idx, thread_idx) to compute the position in the output
    pid = tl.program_id(axis=0)
    num_block_m = tl.num_programs(axis=0)
    num_block_n = tl.num_programs(axis=1)
    block_m = pid // num_block_n
    block_n = pid % num_block_n

    # Compute the block offset in the output
    offs_m = block_m * BLOCK_SIZE
    offs_n = block_n * BLOCK_SIZE
    # Compute the range of indices in the block
    rm = tl.arange(0, BLOCK_SIZE)
    rn = tl.arange(0, BLOCK_SIZE)
    # Compute the offset in the A and B matrices
    # A is (K, M), B is (K, N)
    # So A's stride is (K, M) -> stride_am = M, stride_ak = 1
    # B's stride is (K, N) -> stride_bm = N, stride_bk = 1
    # C is (M, N) -> stride_cm = N, stride_cn = 1

    # Load A and B blocks
    a = tl.load(A + (offs_m + rm) * stride_am + (offs_n + rn) * stride_ak, mask=(offs_m + rm) < M, mask=(offs_n + rn) < N, other=0.0)
    b = tl.load(B + (offs_m + rm) * stride_bm + (offs_n + rn) * stride_bk, mask=(offs_m + rm) < M, mask=(offs_n + rn) < N, other=0.0)
    # Compute the dot product
    c = tl.dot(a, b)
    # Store the result
    tl.store(C + (offs_m + rm) * stride_cm + (offs_n + rn) * stride_cn, c, mask=(offs_m + rm) < M, mask=(offs_n + rn) < N)


def triton_matmul(A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int):
    """
    Perform matrix multiplication using a Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape[0] == K and A.shape[1] == M, "A must have shape (K, M)"
    assert B.shape[0] == K and B.shape[1] == N, "B must have shape (K, N)"

    # Output tensor (M, N)
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Determine block size
    BLOCK_SIZE = 128

    # Compute grid size
    num_block_m = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_block_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_block_m, num_block_n)

    # Launch the kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using a custom Triton kernel.
        """
        return triton_matmul(A, B, A.shape[0], A.shape[1], B.shape[1])