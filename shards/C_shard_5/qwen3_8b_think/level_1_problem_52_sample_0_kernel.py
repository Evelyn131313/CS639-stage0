import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmin(x, dim=self.dim)


def triton_argmin(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    batch_size = x.size(0)
    dim1 = x.size(1)
    dim2 = x.size(2)
    output_size = (batch_size, dim2)
    out = torch.empty(output_size, dtype=torch.int64, device=x.device)

    @triton.jit
    def argmin_kernel(
        x_ptr,  # Pointer to input tensor
        out_ptr,  # Pointer to output tensor
        batch_size: tl.constexpr,
        dim1: tl.constexpr,
        dim2: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Each thread handles a (batch, j) position
        pid = tl.program_id(0)
        batch = pid // dim2
        j = pid % dim2

        # Initialize min_val and min_idx
        min_val = tl.max_float
        min_idx = 0

        # Iterate over i in 0 to dim1-1
        for i in range(dim1):
            offset = batch * dim1 * dim2 + i * dim2 + j
            val = tl.load(x_ptr + offset)
            if val < min_val:
                min_val = val
                min_idx = i

        # Store the result
        out_offset = batch * dim2 + j
        tl.store(out_ptr + out_offset, min_idx)

    # Launch the kernel
    grid = (batch_size * dim2,)
    argmin_kernel[grid](x, out, batch_size, dim1, dim2, BLOCK_SIZE=128)
    return out