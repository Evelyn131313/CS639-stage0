import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    batch_size,
    dim1,
    dim2,
    block_size,
):
    pid = tl.program_id(0)
    B = pid // dim2
    i = pid % dim2
    base_idx = B * dim1 * dim2 + i
    offsets = tl.arange(0, block_size)
    mask = offsets < dim1
    x = tl.load(x_ptr + base_idx + offsets * dim2, mask=mask, other=0.0)
    sum_val = tl.sum(x)
    out_idx = B * dim2 + i
    tl.store(out_ptr + out_idx, sum_val / dim1)

def triton_mean(x: torch.Tensor, batch_size: int, dim1: int, dim2: int):
    assert x.is_cuda and x.contiguous(), "Tensor must be on CUDA and contiguous."
    out = torch.empty((batch_size, dim2), dtype=x.dtype, device=x.device)
    block_size = 128  # Tunable parameter

    grid = lambda meta: ((batch_size * dim2 + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    mean_kernel[grid](x, out, batch_size, dim1, dim2, block_size=block_size)
    return out

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            batch_size = x.size(0)
            dim1 = x.size(1)
            dim2 = x.size(2)
            return triton_mean(x, batch_size, dim1, dim2)
        elif self.dim == 2:
            batch_size = x.size(0)
            dim1 = x.size(1)
            dim2 = x.size(2)
            out = torch.empty((batch_size, dim1), dtype=x.dtype, device=x.device)
            block_size = 128
            grid = lambda meta: ((batch_size * dim1 + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
            @triton.jit
            def mean_kernel_dim2(
                x_ptr,  # Pointer to input tensor
                out_ptr,  # Pointer to output tensor
                batch_size,
                dim1,
                dim2,
                block_size,
            ):
                pid = tl.program_id(0)
                B = pid // dim1
                j = pid % dim1
                base_idx = B * dim1 * dim2 + j
                offsets = tl.arange(0, block_size)
                mask = offsets < dim2
                x = tl.load(x_ptr + base_idx + offsets, mask=mask, other=0.0)
                sum_val = tl.sum(x)
                out_idx = B * dim1 + j
                tl.store(out_ptr + out_idx, sum_val / dim2)
            mean_kernel_dim2[grid](x, out, batch_size, dim1, dim2, block_size=block_size)
            return out
        else:
            return torch.mean(x, dim=self.dim)