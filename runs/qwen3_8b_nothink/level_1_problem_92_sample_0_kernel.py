import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    dim,  # Dimension along which to perform the operation
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the index in the batch
    pid = tl.program_id(0)
    # Compute the offset in the batch
    batch_offset = pid * BLOCK_SIZE
    # Compute the offset in the dimension
    dim_offset = tl.arange(0, BLOCK_SIZE)
    # Compute the offset in the rest of the dimensions
    rest_offset = tl.arange(0, BLOCK_SIZE)

    # Compute the total number of elements in the dimension
    dim_size = tl.load(x_ptr + tl.multiple_indices([0, dim, 0, 0], n_elements, 0, 0, 0), other=0)
    # Compute the number of elements in the rest of the dimensions
    rest_size = n_elements // dim_size

    # Loop over the elements in the dimension
    for i in range(dim_size):
        # Compute the offset in the dimension
        dim_index = i + batch_offset
        # Compute the offset in the rest of the dimensions
        rest_index = tl.arange(0, rest_size)

        # Compute the total offset
        total_offset = dim_index * rest_size + rest_index

        # Compute the index in the input and output
        input_index = total_offset
        output_index = total_offset

        # Load the input value
        x = tl.load(x_ptr + input_index, other=0.0)
        # Compute the cumulative sum
        if i == 0:
            out = x
        else:
            out = tl.load(out_ptr + (dim_index - 1) * rest_size + rest_index, other=0.0) + x
        # Store the result
        tl.store(out_ptr + output_index, out, mask=(dim_index < dim_size) & (rest_index < rest_size))


def triton_exclusive_cumsum(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    exclusive_cumsum_kernel[grid](x, out, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).
    Optimized using a custom Triton kernel.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Perform the exclusive cumulative sum using the Triton kernel
        return triton_exclusive_cumsum(x, self.dim)