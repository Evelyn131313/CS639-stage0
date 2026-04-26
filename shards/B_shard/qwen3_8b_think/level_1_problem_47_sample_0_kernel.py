import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_kernel(
    x_ptr,  # Pointer to input tensor
    y_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread processes a single element in the input
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < batch_size * dim1 * dim2

    # Load input values
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)

    # Compute batch, i, j from offset
    batch = offset // (dim1 * dim2)
    remaining = offset % (dim1 * dim2)
    i = remaining // dim2
    j = remaining % dim2

    # Compute output index
    output_index = batch * dim2 + j

    # Load and accumulate in output
    y = tl.load(y_ptr + output_index, mask=mask, other=0.0)
    y += x

    # Store the result
    tl.store(y_ptr + output_index, y, mask=mask)


def triton_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    batch_size = x.shape[0]
    dim1 = x.shape[1]
    dim2 = x.shape[2]

    # Determine output shape based on reduction dimension
    if dim == 1:
        output_shape = (batch_size, 1, dim2)
    else:
        output_shape = (batch_size, dim1, 1)

    y = torch.empty(output_shape, device=x.device, dtype=x.dtype)

    n_elements = batch_size * dim1 * dim2
    BLOCK_SIZE = 128  # Tunable parameter

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    sum_kernel[grid](x, y, batch_size, dim1, dim2, BLOCK_SIZE=BLOCK_SIZE)
    return y


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum(x, self.dim)