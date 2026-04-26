import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_strides,  # Input strides (NHWC)
    output_strides,  # Output strides (NHWC)
    kernel_size,  # Kernel size (H x W)
    num_channels,  # Number of input channels
    num_output_channels,  # Number of output channels
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    # Compute the 3D thread ID (n, h, w)
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Compute the offset in the input and output
    n = pid_n
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    # Compute the input offset (NHWC)
    input_offset = (
        n * input_strides[0] +
        h * input_strides[1] +
        w * input_strides[2] +
        tl.arange(0, BLOCK_C)
    )

    # Compute the weight offset (OWHC)
    weight_offset = (
        tl.arange(0, BLOCK_O) * num_channels +
        tl.arange(0, BLOCK_C)
    )

    # Compute the output offset (NHWC)
    output_offset = (
        n * output_strides[0] +
        h * output_strides[1] +
        w * output_strides[2] +
        tl.arange(0, BLOCK_O)
    )

    # Load input and weight
    input_values = tl.load(input_ptr + input_offset, mask=(h < 224) & (w < 224) & (tl.arange(0, BLOCK_C) < num_channels), other=0.0)
    weight_values = tl.load(weight_ptr + weight_offset, mask=(tl.arange(0, BLOCK_O) < num_output_channels) & (tl.arange(0, BLOCK_C) < num_channels), other=0.0)

    # Compute the convolution
    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_O), dtype=tl.float32)
    for c in range(0, num_channels, BLOCK_C):
        for o in range(0, num_output_channels, BLOCK_O):
            input_block = input_values[c:c+BLOCK_C]
            weight_block = weight_values[o:o+BLOCK_O]
            # Compute the dot product
            acc += tl.dot(input_block, weight_block)

    # Store the result
    tl.store(output_ptr + output_offset, acc, mask=(h < 224) & (w < 224) & (tl.arange(0, BLOCK_O) < num_output_channels))


def triton_conv2d(input, weight, output):
    assert input.is_cuda and weight.is_cuda and output.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = output.contiguous()

    # Compute strides for NHWC format
    input_strides = (input.stride(0), input.stride(1), input.stride(2))
    output_strides = (output.stride(0), output.stride(1), output.stride(2))

    # Determine block sizes
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_C = 16
    BLOCK_O = 16

    # Compute grid size
    num_output_channels = weight.size(0)
    num_channels = weight.size(1)
    grid = (num_output_channels, 224 // BLOCK_H, 224 // BLOCK_W)

    # Launch the kernel
    conv2d_kernel[grid](
        input, weight, output,
        input_strides, output_strides,
        11, num_channels, num_output_channels,
        BLOCK_H, BLOCK_W, BLOCK_C, BLOCK_O
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(96, 3, 11, 11).cuda())
        self.bias = nn.Parameter(torch.zeros(96).cuda())

    def forward(self, x):
        output = torch.zeros((x.size(0), 96, 56, 56), dtype=torch.float32, device=x.device)
        triton_conv2d(x, self.weight, output)
        return output