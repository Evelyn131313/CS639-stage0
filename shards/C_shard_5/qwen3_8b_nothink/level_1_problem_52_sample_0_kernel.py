import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in the input tensor
    dim: tl.constexpr,  # Dimension along which to compute argmin
    BLOCK_SIZE: tl.constexpr,
):
    # Get the index of the current thread
    pid = tl.program_id(0)
    # Compute the offset for the current thread
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offset < n_elements

    # Load the input values
    x = tl.load(x_ptr + offset, mask=mask, other=tl.max_float)

    # Compute the index of the minimum value along the specified dimension
    # We need to compute the argmin along the dimension, so we need to iterate over the other dimensions
    # For simplicity, we assume the input is 2D (batch, dim1, dim2) and dim is 1
    # So for each batch, we iterate over dim2 and find the index of the minimum in dim1
    # We'll handle this by reshaping the tensor into (batch * dim2, dim1)
    # and then compute the argmin along the last dimension

    # Reshape the tensor to (batch * dim2, dim1)
    # We'll treat the input as a 2D tensor (batch * dim2, dim1)
    # So the number of rows is batch * dim2
    # The number of columns is dim1
    # We'll compute the argmin along the columns (dim1)
    # So for each row, we find the index of the minimum value in the column

    # Compute the number of rows
    num_rows = n_elements // dim1

    # Compute the row indices
    row_indices = tl.arange(0, num_rows)

    # Compute the column indices
    col_indices = tl.arange(0, dim1)

    # Compute the linear index for each element
    linear_indices = row_indices[:, None] * dim1 + col_indices[None, :]

    # Compute the value at each linear index
    values = tl.load(x_ptr + linear_indices, mask=mask, other=tl.max_float)

    # Compute the index of the minimum value
    min_val = tl.min(values, axis=1)
    min_indices = tl.argmin(values, axis=1)

    # Convert the min_indices to the original index in the input tensor
    # The original index is (batch, dim1, dim2) -> (batch * dim2, dim1)
    # So the original index is (row_index // dim2, min_indices, row_index % dim2)
    # We need to reshape the min_indices to (num_rows, 1) and broadcast it
    # Then compute the original index as (row_index // dim2, min_indices, row_index % dim2)

    # Compute the original index
    original_indices = (row_indices // dim2)[:, None] * dim2 + min_indices[None, :]

    # Store the result
    tl.store(out_ptr + offset, original_indices, mask=mask)


def triton_argmin(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    # The output shape is (batch, dim2)
    # So the number of elements is batch * dim2
    out = torch.empty((x.size(0), x.size(2)), dtype=torch.int64, device=x.device)

    # Number of elements in the tensor
    n_elements = x.numel()
    dim1 = x.size(1)
    dim2 = x.size(2)

    # Determine the number of blocks needed
    BLOCK_SIZE = 128  # Tunable parameter for block size
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    argmin_kernel[triton.make_kernel(num_blocks, num_blocks)](x, out, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmin(x, dim=self.dim)