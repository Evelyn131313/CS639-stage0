import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_stride,  # Stride of input tensor
    output_stride,  # Stride of output tensor
    kernel_size: tl.constexpr,
    padding: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the position of the current thread
    pid = tl.program_id(0)
    # Compute the offset in the output tensor
    offset = pid * BLOCK_SIZE
    # Compute the start and end indices in the input tensor
    start = offset * stride - padding
    end = start + kernel_size
    # Compute the number of elements in the window
    num_elements = min(end, input_stride) - start
    # Compute the offset in the input tensor
    input_offset = start
    # Load the input values
    input_vals = tl.load(input_ptr + input_offset, mask=input_offset < input_stride, other=0.0)
    # Compute the average
    avg = tl.sum(input_vals) / num_elements
    # Store the result in the output tensor
    tl.store(output_ptr + offset, avg, mask=offset < output_stride)


def triton_avg_pool(input: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    This function wraps the Triton kernel call for 1D Average Pooling.
    """
    assert input.is_cuda, "Input tensor must be on CUDA."
    input = input.contiguous()
    # Compute output shape
    input_length = input.size(2)
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    # Create output tensor
    output = torch.empty((input.size(0), input.size(1), output_length), dtype=input.dtype, device=input.device)
    output = output.contiguous()
    # Number of elements in the output tensor
    n_elements = output.numel()
    # Tunable block size
    BLOCK_SIZE = 128
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    avg_pool_kernel[grid](input, output, input.stride(2), output.stride(2), kernel_size, padding, stride, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    """
    Optimized model with custom Triton kernel for 1D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool(x, self.kernel_size, self.stride, self.padding)