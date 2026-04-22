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
    # A is (M, K), B is (K, N)
    # So A's stride is (K, 1), B's stride is (N, 1)
    # We need to load A's row and B's column
    # For A: A[offs_m + rm, ...]
    # For B: B[... , offs_n + rn]

    # Load A's row
    a = tl.load(A + offs_m + rm * stride_ak, mask=rm < K, other=0.0)
    # Load B's column
    b = tl.load(B + offs_n + rn * stride_bk, mask=rn < K, other=0.0)

    # Compute the dot product
    c = tl.dot(a, b)
    # Store the result
    tl.store(C + offs_m + rn * stride_cm, c, mask=rn < N)


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
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Matrix dimensions do not match for multiplication."
    out = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Determine the block size
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed along M and N
    num_block_m = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_block_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Define the grid
    grid = (num_block_m * num_block_n, )

    # Launch the Triton kernel
    matmul_kernel[grid](
        A, B, out,
        M, N, K,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bm=B.stride(0), stride_bk=B.stride(1),
        stride_cm=out.stride(0), stride_cn=out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)