import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A, B, C, N, BLOCK_SIZE):
    pid = tl.program_id(0)
    i = pid // BLOCK_SIZE
    j = pid % BLOCK_SIZE
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(0, N, BLOCK_SIZE):
        a = tl.load(A + i * N + k, mask=k < N - k, other=0.0)
        b = tl.load(B + k * N + j, mask=k < N - k, other=0.0)
        acc += a * b
    tl.store(C + i * N + j, acc)

def triton_matmul(A, B):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty_like(A)
    N = A.size(0)
    BLOCK_SIZE = 128
    grid = (N // BLOCK_SIZE + 1, )
    matmul_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)