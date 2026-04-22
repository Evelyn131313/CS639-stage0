import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def min_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    dim1,  # Size of dim1
    dim2,  # Size of dim2
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Map pid to (b, i)
    b = pid // dim2
    i = pid % dim2
    base = b * dim1 * dim2 + i * dim1
    # Process reduction dimension (dim1)
    # Each thread handles a block of size BLOCK_SIZE
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim1
    # Load values
    x = tl.load(x_ptr + base + offsets, mask=mask, other=float('inf'))
    # Compute min
    min_val = tl.min(x, mask=mask)
    # Store result
    out_idx = b * dim2 + i
    tl.store(out_ptr + out_idx, min_val)

def triton_min(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    batch_size, dim1, dim2 = x.shape
    out_shape = (batch_size, dim2) if dim == 1 else (batch_size, dim1)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    num_warps = 4
    BLOCK_SIZE = 128
    # Launch kernel
    min_kernel[triton.cdiv(dim1, BLOCK_SIZE), num_warps](x, out, batch_size, dim1, dim2, BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            return triton_min(x, 1)
        elif self.dim == 2:
            # For dim=2, transpose and apply min kernel
            x = x.transpose(1, 2)
            result = triton_min(x, 1)
            return result.transpose(1, 2)
        else:
            raise ValueError("Unsupported reduction dimension")