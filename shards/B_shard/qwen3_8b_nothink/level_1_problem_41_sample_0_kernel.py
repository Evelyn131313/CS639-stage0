import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_pool1d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    indices_ptr,  # Pointer to indices tensor (if return_indices is True)
    stride: tl.constexpr,
    kernel_size: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    batch_size: tl.constexpr,
    num_features: tl.constexpr,
    sequence_length: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    return_indices: tl.constexpr,
):
    # Each program handles a single batch and feature
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    pid_block = tl.program_id(2)

    # Compute the starting index in the input for this batch and feature
    batch_idx = pid_batch * num_features * sequence_length
    feature_idx = pid_feature * sequence_length
    block_start = pid_block * BLOCK_SIZE

    # Compute the offset in the input tensor
    input_offset = batch_idx + feature_idx + block_start
    input_offsets = input_offset + tl.arange(0, BLOCK_SIZE)

    # Compute the corresponding output position
    # Input shape: (batch_size, num_features, sequence_length)
    # Output shape: (batch_size, num_features, ceil((sequence_length + 2*padding - dilation*(kernel_size-1))/stride))
    # We need to compute the output index for each input position

    # Compute the starting position in the input for the current block
    start = block_start + padding
    # Compute the end position in the input for the current block
    end = start + kernel_size * dilation
    # Compute the output position
    output_pos = (start + (kernel_size - 1) * dilation) // stride

    # Compute the range of input positions to consider for max pooling
    input_range = tl.arange(0, kernel_size)
    input_positions = start + input_range * dilation

    # Initialize max value and index
    max_val = -float('inf')
    max_idx = -1

    # Iterate over the input positions
    for i in range(kernel_size):
        pos = input_positions[i]
        if pos >= sequence_length:
            continue
        input_idx = batch_idx + feature_idx + pos
        val = tl.load(input_ptr + input_idx, mask=(pos < sequence_length), other=-float('inf'))
        if val > max_val:
            max_val = val
            max_idx = pos

    # Compute the output index
    output_idx = pid_batch * num_features + pid_feature
    output_idx = output_idx * sequence_length + output_pos

    # Write the result to output
    tl.store(output_ptr + output_idx, max_val, mask=(output_pos < sequence_length))
    if return_indices:
        tl.store(indices_ptr + output_idx, max_idx, mask=(output_pos < sequence_length))


def triton_max_pool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int, return_indices: bool):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()

    # Compute output shape
    output_sequence_length = (x.size(2) + 2 * padding - dilation * (kernel_size - 1)) // stride
    output_sequence_length = (output_sequence_length + 1) // 1  # Ensure integer division

    # Prepare output tensor
    output = torch.empty(x.size(0), x.size(1), output_sequence_length, dtype=x.dtype, device=x.device)
    indices = torch.empty_like(output, dtype=torch.long) if return_indices else None

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: (
        (x.size(0),),  # Batch dimension
        (x.size(1),),  # Feature dimension
        ((x.size(2) + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    )

    # Launch the Triton kernel
    max_pool1d_kernel[grid](
        x,
        output,
        indices,
        stride,
        kernel_size,
        padding,
        dilation,
        x.size(0),
        x.size(1),
        x.size(2),
        BLOCK_SIZE,
        return_indices
    )
    return output, indices if return_indices else output


class ModelNew(nn.Module):
    """
    Optimized Max Pooling 1D layer using custom Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices
        )