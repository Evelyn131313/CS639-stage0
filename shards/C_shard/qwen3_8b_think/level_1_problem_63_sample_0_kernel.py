import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor (batch, in_channels, height, width)
    weight_ptr,  # Pointer to weight tensor (out_channels, in_channels, kernel_size, kernel_size)
    bias_ptr,  # Pointer to bias tensor (out_channels)
    output_ptr,  # Pointer to output tensor (batch, out_channels, out_height, out_width)
    batch: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a single output position (i, j)
    pid = tl.program_id(0)
    i = pid // out_width
    j = pid % out_width

    # Compute input indices for the current output position (i, j)
    input_i = i * stride - padding
    input_j = j * stride - padding

    # Compute the sum for the current output position
    sum_val = tl.zeros((out_channels,), dtype=tl.float32)

    # Iterate over all output channels
    for c in range(out_channels):
        # Iterate over all input channels
        for g in range(in_channels):
            # Iterate over kernel elements
            for k in range(kernel_size):
                for l in range(kernel_size):
                    # Compute input indices for kernel element (k, l)
                    input_i_k = input_i + k
                    input_j_l = input_j + l

                    # Check if the input indices are within bounds
                    if (input_i_k >= 0) and (input_i_k < height) and (input_j_l >= 0) and (input_j_l < width):
                        # Compute the input pointer offset
                        input_offset = (
                            pid * in_channels * height * width
                            + g * height * width
                            + input_i_k * width
                            + input_j_l
                        )
                        input_val = tl.load(input_ptr + input_offset)

                        # Compute the weight pointer offset
                        weight_offset = (
                            c * in_channels * kernel_size * kernel_size
                            + g * kernel_size * kernel_size
                            + k * kernel_size
                            + l
                        )
                        weight_val = tl.load(weight_ptr + weight_offset)

                        # Accumulate the product
                        sum_val[c] += weight_val * input_val

    # Add bias
    for c in range(out_channels):
        bias_val = tl.load(bias_ptr + c)
        sum_val[c] += bias_val

    # Store the result
    for c in range(out_channels):
        output_offset = (
            pid * out_channels
            + c
        )
        tl.store(output_ptr + output_offset, sum_val[c])


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Prepare output tensor
    batch, in_channels, height, width = input.shape
    out_channels, _, kernel_size, _ = weight.shape
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    output = torch.empty((batch, out_channels, out_height, out_width), dtype=input.dtype, device=input.device)

    # Number of elements in the output
    n_elements = batch * out_channels * out_height * out_width
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, bias, output, batch, in_channels, out_channels, kernel_size, stride, padding, height, width, out_height, out_width, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure the weights and bias are initialized
        if not hasattr(self, "weight"):
            self.weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size).cuda())
        if self.bias:
            if not hasattr(self, "bias"):
                self.bias = torch.nn.Parameter(torch.randn(self.out_channels).cuda())
        
        # Perform the convolution using Triton kernel
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding)