import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def frobenius_norm_kernel(
    x_ptr,  # pointer to input tensor
    norm_ptr,  # pointer to output norm
    n_elements,  # number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_sq = x * x
    block_sum = tl.sum(x_sq, axis=0)

    # Accumulate block_sum into shared memory
    shared = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    shared[pid] = block_sum
    tl.debug_barrier()

    # Sum all block sums
    total_sum = tl.sum(shared)
    tl.store(norm_ptr, total_sum)

def compute_norm(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    n_elements = x.numel()
    BLOCK_SIZE = 128
    norm = torch.tensor(0.0, device=x.device)
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    frobenius_norm_kernel[grid](x, norm, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return torch.sqrt(norm)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = compute_norm(x)
        return x / norm