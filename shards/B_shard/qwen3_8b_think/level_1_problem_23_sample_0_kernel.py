import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr, 
    out_ptr, 
    batch_size, 
    dim, 
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    offsets = row_idx * dim + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * dim
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    max_val = tl.max(x)
    x_exp = tl.exp(x - max_val)
    sum_exp = tl.sum(x_exp)
    out = x_exp / sum_exp
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_softmax(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size = x.size(0)
    row_size = x.size(1)
    BLOCK_SIZE = row_size
    grid = (batch_size, 1)
    softmax_kernel[grid, (BLOCK_SIZE, 1)](x, out, batch_size, row_size, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax(x, dim=1)