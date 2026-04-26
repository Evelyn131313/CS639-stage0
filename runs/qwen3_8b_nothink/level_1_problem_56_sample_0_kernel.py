import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_strides,  # Input strides (N, C, H, W)
    output_strides,  # Output strides (N, C, H, W)
    kernel_size_h, kernel_size_w,  # Kernel size
    stride_h, stride_w,  # Stride
    padding_h, padding_w,  # Padding
    dilation_h, dilation_w,  # Dilation
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the position in the output
    pid = tl.program_id(0)
    # Compute the output position (n, oc, oh, ow)
    oh = pid // (output_strides[2] * output_strides[3])
    ow = pid % (output_strides[3])
    oc = (pid // output_strides[3]) % output_strides[2]
    n = pid // (output_strides[2] * output_strides[3] * output_strides[1])

    # Compute the output offset
    output_offset = n * output_strides[0] + oc * output_strides[1] + oh * output_strides[2] + ow * output_strides[3]
    output_ptr += output_offset

    # Compute the input offset for the top-left corner of the kernel
    input_h_start = oh * stride_h - padding_h
    input_w_start = ow * stride_w - padding_w

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Iterate over the kernel
    for kh in range(kernel_size_h):
        for kw in range(kernel_size_w):
            # Compute the input offset for this kernel position
            input_h = input_h_start + kh * dilation_h
            input_w = input_w_start + kw * dilation_w

            # Compute the input offset
            input_offset = n * input_strides[0] + oc * input_strides[1] + input_h * input_strides[2] + input_w * input_strides[3]
            input_ptr += input_offset

            # Load the weight
            weight_offset = oc * input_strides[1] + kh * input_strides[2] + kw * input_strides[3]
            weight_ptr += weight_offset
            weight = tl.load(weight_ptr, mask=tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE, other=0.0)
            weight_ptr -= weight_offset

            # Load the input
            input_val = tl.load(input_ptr, mask=tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE, other=0.0)
            input_ptr -= input_offset

            # Multiply and accumulate
            acc += input_val * weight

    # Store the result
    tl.store(output_ptr, acc, mask=tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE)


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: tuple, padding: tuple, dilation: tuple):
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

    # Compute output shape
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation

    out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    output = torch.empty((batch_size, out_channels, out_height, out_width), dtype=input.dtype, device=input.device)

    # Compute input and output strides
    input_strides = (input.stride(0), input.stride(1), input.stride(2), input.stride(3))
    output_strides = (output.stride(0), output.stride(1), output.stride(2), output.stride(3))

    # Number of elements in the output
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, input_strides, output_strides, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on GPU
        x = x.cuda()
        # Create weight and bias tensors
        weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels // self.groups, *self.kernel_size, device=x.device))
        if self.bias:
            bias = torch.nn.Parameter(torch.randn(self.out_channels, device=x.device))
        else:
            bias = None
        # Perform convolution using Triton kernel
        return triton_conv2d(x, weight, bias, self.stride, self.padding, self.dilation)