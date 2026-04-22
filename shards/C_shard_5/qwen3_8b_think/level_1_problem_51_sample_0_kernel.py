import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmax(x, self.dim)

def triton_argmax(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    batch_size = x.size(0)
    dim1 = x.size(1)
    dim2 = x.size(2)
    out = torch.empty(batch_size, dim2, dtype=torch.int64, device=x.device)

    @triton.jit
    def argmax_kernel(
        x_ptr,  # Pointer to input tensor
        out_ptr,  # Pointer to output tensor
        batch_size,  # Number of batches
        dim1,  # Dimension to reduce over
        dim2,  # Other dimension
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch = pid // dim2
        j = pid % dim2

        start = batch * dim1 * dim2 + j
        max_val = -float('inf')
        max_index = -1

        # Iterate over all elements in the reduction dimension
        for i in range(dim1):
            idx = start + i * dim2
            val = tl.load(x_ptr + idx, mask=idx < x_ptr.size, other=-float('inf'))
            if val > max_val:
                max_val = val
                max_index = i

        out_idx = batch * dim2 + j
        tl.store(out_ptr + out_idx, max_index)

    grid = (batch_size * dim2,)
    argmax_kernel[grid](x, out, batch_size, dim1, dim2, BLOCK_SIZE=128)
    return out