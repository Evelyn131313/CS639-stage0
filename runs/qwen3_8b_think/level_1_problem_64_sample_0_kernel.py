import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, length = x.shape
        out_channels = self.out_channels
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        output_padding = self.output_padding
        groups = self.groups
        bias = self.bias

        # Calculate output length
        output_length = (length - 1) * stride + kernel_size - 2 * padding + output_padding
        out = torch.empty((batch_size, out_channels, output_length), device=x.device, dtype=x.dtype)

        # Prepare weights and bias
        weights = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, device=x.device, dtype=x.dtype))
        if bias:
            bias_tensor = torch.nn.Parameter(torch.randn(out_channels, device=x.device, dtype=x.dtype))
        else:
            bias_tensor = None

        # Launch Triton kernel
        self.triton_transposed_conv(x, weights, bias_tensor, out, length, kernel_size, stride, padding, output_padding)
        return out

    @torch.no_grad()
    def triton_transposed_conv(self, x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, out: torch.Tensor, input_length: int, kernel_size: int, stride: int, padding: int, output_padding: int):
        assert x.is_cuda and w.is_cuda and (bias is None or bias.is_cuda) and out.is_cuda, "Tensors must be on CUDA."
        x = x.contiguous()
        w = w.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        out = out.contiguous()

        # Calculate output length
        output_length = (input_length - 1) * stride + kernel_size - 2 * padding + output_padding
        n_output_elements = output_length

        # Determine the number of blocks needed
        BLOCK_SIZE = 128
        grid = lambda meta: ((n_output_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

        # Launch the Triton kernel
        transposed_conv_kernel[grid](x, w, bias, out, input_length, n_output_elements, kernel_size, stride, padding, output_padding, BLOCK_SIZE=BLOCK_SIZE)

@triton.jit
def transposed_conv_kernel(
    x_ptr,  # input tensor
    w_ptr,  # weights tensor
    bias_ptr,  # bias tensor
    out_ptr,  # output tensor
    input_length: tl.constexpr,
    n_output_elements: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_output_elements

    # For each offset i (output element)
    for i in range(BLOCK_SIZE):
        if mask[i]:
            # Compute the sum of the input elements multiplied by the kernel weights
            sum_val = 0.0
            # For each kernel index k
            for k in range(kernel_size):
                # Compute the input index
                input_index = i - k
                # Check if input_index is within bounds
                if input_index >= 0 and input_index < input_length:
                    # Load the input value
                    x_val = tl.load(x_ptr + input_index, mask=..., other=0.0)
                else:
                    x_val = 0.0
                # Load the weight
                w_val = tl.load(w_ptr + k, mask=..., other=0.0)
                sum_val += x_val * w_val
            # Add bias if present
            if bias_ptr is not None:
                sum_val += tl.load(bias_ptr, mask=..., other=0.0)
            # Store the result
            tl.store(out_ptr + i, sum_val, mask=...)