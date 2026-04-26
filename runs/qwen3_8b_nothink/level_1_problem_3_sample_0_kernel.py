import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    A_ptr,  # Pointer to A
    B_ptr,  # Pointer to B
    C_ptr,  # Pointer to C
    batch_size: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the batch index
    batch_idx = pid // (BLOCK_M * BLOCK_N)
    # Compute the block index within the batch
    block_idx_m = (pid % (BLOCK_M * BLOCK_N)) // BLOCK_N
    block_idx_n = (pid % (BLOCK_N))  # Only one block in N direction

    # Compute the offset in the batch
    batch_offset = batch_idx * m * k * n

    # Compute the offset in A and B
    A_offset = batch_offset + block_idx_m * k * m + tl.arange(0, BLOCK_M) * k + tl.arange(0, BLOCK_K)
    B_offset = batch_offset + block_idx_n * n * k + tl.arange(0, BLOCK_K) * n + tl.arange(0, BLOCK_N)

    # Load A and B
    A = tl.load(A_ptr + A_offset, mask=tl.arange(0, BLOCK_M) < m, other=0.0)
    B = tl.load(B_ptr + B_offset, mask=tl.arange(0, BLOCK_N) < n, other=0.0)

    # Compute the dot product
    C = tl.dot(A, B)

    # Store the result
    C_offset = batch_offset + block_idx_m * n * m + tl.arange(0, BLOCK_M) * n + tl.arange(0, BLOCK_N)
    tl.store(C_ptr + C_offset, C, mask=tl.arange(0, BLOCK_M) < m & tl.arange(0, BLOCK_N) < n)


def triton_bmm(A: torch.Tensor, B: torch.Tensor):
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
    C = torch.empty((A.size(0), A.size(1), B.size(2)), dtype=A.dtype, device=A.device)

    # Parameters
    batch_size = A.size(0)
    m = A.size(1)
    n = B.size(2)
    k = A.size(2)

    # Tunable parameters
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    # Compute the number of blocks needed
    num_blocks_m = (m + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (n + BLOCK_N - 1) // BLOCK_N
    num_blocks = num_blocks_m * num_blocks_n

    # Launch the Triton kernel
    bmm_kernel[ num_blocks ](A, B, C, batch_size, m, n, k, BLOCK_M, BLOCK_N, BLOCK_K)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_bmm(A, B)