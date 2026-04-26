import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block corresponds to a (batch, j) pair
    block_idx = tl.program_id(0)
    batch = block_idx // dim2
    j = block_idx % dim2

    # Each thread in the block corresponds to a k in dim1
    thread_idx = tl.program_id(1)
    if thread_idx >= dim1:
        return

    # Compute the offset for the current (batch, j, k)
    offset = batch * dim1 * dim2 + j * dim1 + thread_idx
    x = tl.load(x_ptr + offset, mask=offset < x_ptr.size, other=0.0)

    # Shared memory for min_val and min_index
    # We'll use a shared memory array of size 2 for each thread, but this is not feasible
    # So, we'll use a single shared memory variable for min_val and min_index

    # Shared memory for min_val and min_index
    shared_min_val = tl.shared(float, 1)
    shared_min_index = tl.shared(int, 1)

    # Initialize shared memory
    if thread_idx == 0:
        shared_min_val[0] = float('inf')
        shared_min_index[0] = -1

    # Synchronize to ensure shared_min_val and shared_min_index are initialized
    tl.sync()

    # Compare current value with shared min
    if x < shared_min_val[0]:
        shared_min_val[0] = x
        shared_min_index[0] = thread_idx

    # Synchronize to ensure all threads have updated the shared values
    tl.sync()

    # Write the result to the output
    out_offset = batch * dim2 + j
    tl.store(out_ptr + out_offset, shared_min_index[0], mask=out_offset < out_ptr.size)


def triton_argmin(x: torch.Tensor, dim: int):
    assert x.is_cuda and dim == 1, "Tensors must be on CUDA and dim must be 1."
    x = x.contiguous()
    out = torch.empty((x.size(0), x.size(2)), device=x.device, dtype=torch.int64)

    batch_size = x.size(0)
    dim1 = x.size(1)
    dim2 = x.size(2)

    # Number of blocks
    num_blocks = batch_size * dim2

    # Launch the kernel
    grid = (num_blocks, 1)
    argmin_kernel[grid](x, out, batch_size, dim1, dim2, BLOCK_SIZE=dim1)

    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmin(x, self.dim)