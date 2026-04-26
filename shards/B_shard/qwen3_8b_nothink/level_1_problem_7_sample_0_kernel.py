import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a 16x16 block of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Compute the block position in C
    block_m = pid_m * BLOCK_SIZE
    block_n = pid_n * BLOCK_SIZE
    # Compute the block offset in C
    offs_cm = block_m + tl.arange(0, BLOCK_SIZE)
    offs_cn = block_n + tl.arange(0, BLOCK_SIZE)
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Iterate over the K dimension
    for k in range(0, K, BLOCK_SIZE):
        # Load A and B
        a = tl.load(A + (offs_cm[:, None] * stride_am + k * stride_ak), mask=(offs_cm < M)[:, None], other=0.0)
        b = tl.load(B + (k * stride_bm + offs_cn[None, :]), mask=(offs_cn < N), other=0.0)
        # Compute the dot product
        acc += tl.dot(a, b)
    # Write back the result
    tl.store(C + (offs_cm[:, None] * stride_cm + offs_cn[None, :]), acc, mask=(offs_cm < M)[:, None] & (offs_cn < N)[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Custom matrix multiplication using Triton.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    # Ensure the tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    # Compute the output tensor
    C = torch.empty((A.shape[0], B.shape[1]), dtype=A.dtype, device=A.device)
    # Define the block size
    BLOCK_SIZE = 16
    # Compute the grid dimensions
    grid = (triton.cdiv(A.shape[0], BLOCK_SIZE), triton.cdiv(B.shape[1], BLOCK_SIZE))
    # Launch the kernel
    matmul_kernel[grid](A, B, C,
                        A.shape[0], B.shape[1], B.shape[0],
                        A.stride(0), A.stride(1),
                        B.stride(0), B.stride(1),
                        C.stride(0), C.stride(1),
                        BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A.cuda(), B.cuda())