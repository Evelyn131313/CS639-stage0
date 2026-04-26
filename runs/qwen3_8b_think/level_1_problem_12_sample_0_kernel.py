import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_mult_kernel(
    A_ptr,  # Pointer to A (shape N)
    B_ptr,  # Pointer to B (shape N, M)
    out_ptr,  # Pointer to output (shape N, M)
    N: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the number of blocks per row
    num_blocks_per_row = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Program id is the global index in the grid
    pid = tl.program_id(0)
    # Determine the row index i and block index within the row
    i = pid // num_blocks_per_row
    block_idx = pid % num_blocks_per_row
    # Each thread in the block handles a column j in the block
    j = tl.arange(0, BLOCK_SIZE)
    # Compute the start column for this block
    start_col = block_idx * BLOCK_SIZE
    # Compute the index in B: (i, j + start_col)
    idx = i * M + (j + start_col)
    # Load A[i]
    a = tl.load(A_ptr + i, mask=i < N, other=0.0)
    # Load B[i, j + start_col]
    b = tl.load(B_ptr + idx, mask=idx < N*M, other=0.0)
    # Compute the product
    out = a * b
    # Store the result in out_ptr
    tl.store(out_ptr + idx, out, mask=idx < N*M)


def diag_mult(A: torch.Tensor, B: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    # Prepare output tensor
    out = torch.empty_like(B)

    # Kernel parameters
    N = A.shape[0]
    M = B.shape[1]
    BLOCK_SIZE = 128  # Choose a suitable block size, e.g., 128
    num_blocks_per_row = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = N * num_blocks_per_row

    # Launch the kernel
    diag_mult_kernel[grid](A, B, out, N, M, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return diag_mult(A, B)