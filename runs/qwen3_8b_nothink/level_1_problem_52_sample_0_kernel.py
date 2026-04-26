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
    # and then compute the argmin along the second dimension

    # Reshape the tensor to (batch * dim2, dim1)
    # We'll treat the input as a 2D tensor (batch * dim2, dim1)
    # So the number of rows is batch * dim2
    # The number of columns is dim1
    # We'll compute the argmin along the columns (dim1)
    # So for each row, we find the index of the minimum value in the column

    # Compute the number of rows
    num_rows = n_elements // dim1

    # Compute the row index for the current thread
    row_idx = pid // dim2
    # Compute the column index for the current thread
    col_idx = pid % dim2

    # Load the values for the current row
    x_row = tl.load(x_ptr + row_idx * dim1 + col_idx, mask=col_idx < dim1, other=tl.max_float)

    # Find the index of the minimum value in the row
    min_val = tl.max_float
    min_idx = 0
    for i in range(dim1):
        val = tl.load(x_ptr + row_idx * dim1 + i, mask=i < dim1, other=tl.max_float)
        if val < min_val:
            min_val = val
            min_idx = i

    # Store the result
    tl.store(out_ptr + row_idx, min_idx, mask=row_idx < num_rows)


def triton_argmin(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()

    # Compute the output shape
    # For dim=1, the output shape is (batch, dim2)
    # For dim=0, the output shape is (dim1, dim2)
    # For dim=2, the output shape is (batch, dim1)
    # We assume dim is 1 as per the example
    # So the output shape is (batch, dim2)
    output_shape = list(x.shape)
    output_shape[dim] = 1
    output_shape = tuple(output_shape)

    # Prepare output tensor
    out = torch.empty(output_shape, device=x.device, dtype=torch.int64)

    # Number of elements in the tensor
    n_elements = x.numel()
    dim1 = x.shape[1]
    dim2 = x.shape[2]

    # Determine the number of blocks needed
    # We use a block size of 128 for the row index
    BLOCK_SIZE = 128
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    argmin_kernel[triton.next_power_of_two(num_blocks)](x, out, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmin(x, self.dim)