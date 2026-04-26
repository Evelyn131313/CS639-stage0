import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Initialize kernel weights
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.reset_parameters()

        # Initialize bias if needed
        if self.bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        # Initialize weights with random values
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _flip_kernel(self, kernel):
        # Flip kernel both vertically and horizontally
        flipped_kernel = kernel.flip(2).flip(3)
        return flipped_kernel

    @triton.jit
    def conv_transpose2d_kernel(
        input_ptr,  # Pointer to input tensor
        weight_ptr,  # Pointer to weight tensor
        bias_ptr,  # Pointer to bias tensor
        output_ptr,  # Pointer to output tensor
        batch_size: tl.constexpr,
        in_channels: tl.constexpr,
        out_channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        kernel_h: tl.constexpr,
        kernel_w: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        padding_h: tl.constexpr,
        padding_w: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Each program handles a single output position
        pid = tl.program_id(0)
        # Compute the output position (i, j)
        i = pid // width
        j = pid % width

        # Compute the input position range
        input_i_start = i - kernel_h + 1
        input_j_start = j - kernel_w + 1

        # Compute the input indices with padding
        input_i_start += padding_h
        input_j_start += padding_w

        # Compute the input offset
        input_offset = (i * width + j) * in_channels * height * width
        output_offset = (i * width + j) * out_channels

        # Load the kernel weights
        kernel = tl.load(weight_ptr, shape=(out_channels, in_channels, kernel_h, kernel_w))

        # Initialize the output value
        output = tl.zeros((out_channels,), dtype=tl.float32)

        # Iterate over the kernel indices
        for k in range(kernel_h):
            for l in range(kernel_w):
                # Compute the input position
                input_i = input_i_start + k
                input_j = input_j_start + l

                # Compute the input offset
                input_offset_k = (input_i * width + input_j) * in_channels

                # Load the input values
                input_val = tl.load(input_ptr + input_offset_k, dtype=tl.float32)

                # Multiply by the kernel weight
                kernel_val = kernel[k][l]
                output += input_val * kernel_val

        # Add bias if present
        if bias_ptr is not None:
            bias = tl.load(bias_ptr, shape=(out_channels,))
            output += bias

        # Store the output value
        tl.store(output_ptr + output_offset, output)

    def triton_conv_transpose2d(self, x: torch.Tensor):
        # Ensure input is on GPU
        assert x.is_cuda, "Input tensor must be on CUDA."

        # Calculate output dimensions
        out_h = (x.size(2) - 1) * self.stride[0] + self.kernel_size[0] - 2 * self.padding[0]
        out_w = (x.size(3) - 1) * self.stride[1] + self.kernel_size[1] - 2 * self.padding[1]

        # Prepare output tensor
        output = torch.empty((x.size(0), self.out_channels, out_h, out_w), dtype=x.dtype, device=x.device)

        # Launch the Triton kernel
        grid = (x.size(0) * out_h * out_w,)
        self.conv_transpose2d_kernel[grid](
            x, self.weight, self.bias, output,
            x.size(0), x.size(1), self.out_channels,
            x.size(2), x.size(3),
            self.kernel_size[0], self.kernel_size[1],
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            128  # BLOCK_SIZE
        )
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.triton_conv_transpose2d(x)