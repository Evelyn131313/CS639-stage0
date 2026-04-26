import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread handles one output element
    pid = tl.program_id(0)
    # Compute the output position (n, c, h, w)
    n = pid // (out_channels * height * width)
    c = (pid // (height * width)) % out_channels
    h = (pid // width) % height
    w = pid % width

    # Compute the input position (n, c, h, w)
    # For depthwise conv, input channels = output channels
    input_c = c
    input_h_start = h - padding
    input_w_start = w - padding

    # Compute the range of kernel positions
    kernel_h = tl.arange(0, kernel_size)
    kernel_w = tl.arange(0, kernel_size)

    # Compute the input indices with dilation
    input_h = input_h_start + kernel_h * dilation
    input_w = input_w_start + kernel_w * dilation

    # Mask to ensure we stay within bounds
    mask_h = (input_h >= 0) & (input_h < height)
    mask_w = (input_w >= 0) & (input_w < width)
    mask = mask_h & mask_w

    # Initialize output
    acc = tl.zeros((kernel_size, kernel_size), dtype=tl.float32)

    # Load weights
    weight = tl.load(weight_ptr + c * kernel_size * kernel_size + kernel_h * kernel_size + kernel_w, mask=mask, other=0.0)

    # Iterate over input channels (only one for depthwise)
    for input_h_idx in range(kernel_size):
        for input_w_idx in range(kernel_size):
            if mask[input_h_idx, input_w_idx]:
                input_idx = n * in_channels * height * width + input_c * height * width + input_h_idx * width + input_w_idx
                input_val = tl.load(input_ptr + input_idx, other=0.0)
                acc[input_h_idx, input_w_idx] += input_val * weight[input_h_idx, input_w_idx]

    # Store the result
    output_idx = n * out_channels * height * width + c * height * width + h * width + w
    tl.store(output_ptr + output_idx, acc.sum())


def triton_depthwise_conv2d(input: torch.Tensor, weight: torch.Tensor, batch_size: int, in_channels: int, out_channels: int, height: int, width: int, kernel_size: int, stride: int, padding: int, dilation: int):
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty((batch_size, out_channels, height, width), dtype=input.dtype, device=input.device)

    # Compute the number of output elements
    n_elements = batch_size * out_channels * height * width
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    depthwise_conv2d_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, height, width, kernel_size, stride, padding, dilation, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize weights and bias
        self.weight = torch.nn.Parameter(torch.randn(in_channels, kernel_size, kernel_size, device='cuda', dtype=torch.float32))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(in_channels, device='cuda', dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform the depthwise 2D convolution using the Triton kernel
        output = triton_depthwise_conv2d(x, self.weight, x.size(0), self.in_channels, self.in_channels, x.size(2), x.size(3), self.kernel_size, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output