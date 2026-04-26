import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A, B, C, n, m, k, BLOCK_SIZE):
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j = tl.arange(0, BLOCK_SIZE)
    # Load A[i][p] and B[p][j]
    p = tl.arange(0, BLOCK_SIZE)
    A_block = tl.load(A + i[:, None] * k + p[None, :], mask=p < k, other=0.0)
    B_block = tl.load(B + p[:, None] * m + j[None, :], mask=p < k, other=0.0)
    C_block = tl.dot(A_block, B_block)
    tl.store(C + i[:, None] * m + j[None, :], C_block)

def triton_matmul(A, B):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    n, m = A.shape
    k = B.shape[1]
    out = torch.empty((n, m), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    matmul_kernel[grid](A, B, out, n, m, k, BLOCK_SIZE=128)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)