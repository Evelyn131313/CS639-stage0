import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to A (K x M)
    B_ptr,  # Pointer to B (K x N)
    out_ptr,  # Pointer to output (M x N)
    M: tl.constexpr,  # M dimension
    K: tl.constexpr,  # K dimension
    N: tl.constexpr,  # N dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread block processes a single element of the output matrix
    pid = tl.program_id(0)
    i = pid // N
    j = pid % N
    # Compute the sum over k
    sum_val = 0.0
    for k in range(K):
        a = tl.load(A_ptr + k * M + i, other=0.0)
        b = tl.load(B_ptr + k * N + j, other=0.0)
        sum_val += a * b
    tl.store(out_ptr + i * N + j, sum_val)

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    out = torch.empty((A.shape[1], B.shape[1]), device=A.device, dtype=A.dtype)
    M = A.shape[1]
    K = A.shape[0]
    N = B.shape[1]
    BLOCK_SIZE = 128
    grid = (M * N, )
    matmul_kernel[grid](A, B, out, M, K, N, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)