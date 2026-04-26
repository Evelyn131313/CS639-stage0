import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, length = x.shape
        output_length = (length - 1) * self.stride + self.kernel_size - 2 * self.padding
        output = torch.empty(batch_size, self.out_channels, output_length, dtype=x.dtype, device=x.device)
        self._transposed_conv(x, output)
        return output

    def _transposed_conv(self, input, output):
        assert input.is_cuda and output.is_cuda, "Tensors must be on CUDA."
        input = input.contiguous()
        output = output.contiguous()

        # Calculate the number of elements
        n_elements = output.numel()
        BLOCK_SIZE = 128

        # Determine the number of blocks needed
        grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

        # Launch the Triton kernel
        transposed_conv_kernel[grid](input, self.weight, self.bias, output, 
                                    self.in_channels, self.out_channels, self.kernel_size, 
                                    self.stride, self.dilation, self.padding, 
                                    input.shape[1], input.shape[2], output.shape[1], output.shape[2], 
                                    BLOCK_SIZE)

@triton.jit
def transposed_conv_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    output_ptr,  # Pointer to output tensor
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    dilation: tl.constexpr,
    padding: tl.constexpr,
    input_channels: tl.constexpr,
    input_length: tl.constexpr,
    output_channels: tl.constexpr,
    output_length: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    # Get the output element index
    out_pos = pid

    # Check if out_pos is within the output length
    if out_pos >= output_length:
        return

    # For each batch
    for batch in range(input_channels):
        # For each output channel
        for out_channel in range(output_channels):
            # Initialize the output value
            out_val = 0.0

            # For each input channel
            for in_channel in range(in_channels):
                # For each kernel index
                for i in range(kernel_size):
                    # Compute input_pos
                    input_pos = out_pos - i * stride * dilation

                    # Check if input_pos is within the input length
                    if input_pos < 0 or input_pos >= input_length:
                        input_val = 0.0
                    else:
                        # Load the input value
                        input_val = tl.load(input_ptr + batch * in_channels * input_length + in_channel * input_length + input_pos)

                    # Load the weight value
                    weight_val = tl.load(weight_ptr + out_channel * in_channels * kernel_size + in_channel * kernel_size + i)

                    # Multiply and accumulate
                    out_val += input_val * weight_val

            # Add bias if present
            if bias_ptr is not None:
                bias_val = tl.load(bias_ptr + out_channel)
                out_val += bias_val

            # Store the output value
            tl.store(output_ptr + batch * output_channels * output_length + out_channel * output_length + out_pos, out_val)