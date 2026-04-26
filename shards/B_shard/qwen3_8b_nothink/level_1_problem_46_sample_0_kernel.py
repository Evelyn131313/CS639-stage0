import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool3d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch, channels, depth, height, width)
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the index in the output tensor
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    # Calculate the position in the output
    out_idx = offset + tl.arange(0, BLOCK_SIZE)
    # Convert output index to input index
    # Output shape is (batch, channels, out_depth, out_height, out_width)
    # Input shape is (batch, channels, in_depth, in_height, in_width)
    # out_depth = (in_depth + 2*padding - kernel_size) // stride + 1
    # out_height = (in_height + 2*padding - kernel_size) // stride + 1
    # out_width = (in_width + 2*padding - kernel_size) // stride + 1

    # For each output position, compute the corresponding input region
    # and compute the average over the kernel
    for out_idx in range(BLOCK_SIZE):
        # Extract the output index
        out_batch, out_channel, out_depth, out_height, out_width = tl.to_int(out_idx)
        # Compute the input indices
        in_depth_start = out_depth * stride - padding
        in_depth_end = in_depth_start + kernel_size
        in_height_start = out_height * stride - padding
        in_height_end = in_height_start + kernel_size
        in_width_start = out_width * stride - padding
        in_width_end = in_width_start + kernel_size

        # Ensure the input indices are within bounds
        in_depth_start = tl.maximum(in_depth_start, 0)
        in_depth_end = tl.minimum(in_depth_end, input_shape[2])
        in_height_start = tl.maximum(in_height_start, 0)
        in_height_end = tl.minimum(in_height_end, input_shape[3])
        in_width_start = tl.maximum(in_width_start, 0)
        in_width_end = tl.minimum(in_width_end, input_shape[4])

        # Compute the number of elements in the kernel
        num_elements = (in_depth_end - in_depth_start) * (in_height_end - in_height_start) * (in_width_end - in_width_start)
        if num_elements == 0:
            continue

        # Compute the sum over the kernel
        sum_val = 0.0
        for d in range(in_depth_start, in_depth_end):
            for h in range(in_height_start, in_height_end):
                for w in range(in_width_start, in_width_end):
                    idx = (out_batch, out_channel, d, h, w)
                    val = tl.load(input_ptr + idx, mask=idx[2] < input_shape[2], other=0.0)
                    sum_val += val

        # Compute the average
        avg = sum_val / num_elements
        # Store the result
        tl.store(output_ptr + out_idx, avg)


def triton_avg_pool3d(input: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    This function wraps the Triton kernel call for 3D average pooling.
    """
    assert input.is_cuda, "Input tensor must be on CUDA."
    input = input.contiguous()
    output_shape = (
        input.shape[0],
        input.shape[1],
        (input.shape[2] + 2 * padding - kernel_size) // stride + 1,
        (input.shape[3] + 2 * padding - kernel_size) // stride + 1,
        (input.shape[4] + 2 * padding - kernel_size) // stride + 1
    )
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Prepare input shape as a tuple
    input_shape = (input.shape[0], input.shape[1], input.shape[2], input.shape[3], input.shape[4])

    # Determine the number of blocks needed
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Launch the Triton kernel
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    avg_pool3d_kernel[grid](input, output, input_shape, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for 3D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)