import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Calculate output shape
        self.depth_out = ((self.padding[0] * 2) + self.kernel_size[0] - self.stride[0] + self.output_padding[0])
        self.height_out = ((self.padding[1] * 2) + self.kernel_size[1] - self.stride[1] + self.output_padding[1])
        self.width_out = ((self.padding[2] * 2) + self.kernel_size[2] - self.stride[2] + self.output_padding[2])

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output shape
        batch_size, in_channels, depth, height, width = x.shape
        out_channels = self.out_channels
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        output_padding = self.output_padding

        # Compute output dimensions
        depth_out = ((depth - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0])
        height_out = ((height - 1) * stride[1] - 2 * padding[1] + kernel_size[1] + output_padding[1])
        width_out = ((width - 1) * stride[2] - 2 * padding[2] + kernel_size[2] + output_padding[2])

        # Initialize output
        output = torch.empty((batch_size, out_channels, depth_out, height_out, width_out), device=x.device, dtype=x.dtype)

        # Launch the Triton kernel
        self._triton_conv_transpose3d(x, output)

        return output

    @triton.jit
    def _triton_conv_transpose3d_kernel(
        x_ptr,  # Pointer to input tensor
        weight_ptr,  # Pointer to weight tensor
        bias_ptr,  # Pointer to bias tensor
        output_ptr,  # Pointer to output tensor
        batch_size,  # Number of batches
        in_channels,  # Input channels
        out_channels,  # Output channels
        kernel_depth,  # Kernel depth
        kernel_height,  # Kernel height
        kernel_width,  # Kernel width
        stride_depth,  # Stride depth
        stride_height,  # Stride height
        stride_width,  # Stride width
        padding_depth,  # Padding depth
        padding_height,  # Padding height
        padding_width,  # Padding width
        output_padding_depth,  # Output padding depth
        output_padding_height,  # Output padding height
        output_padding_width,  # Output padding width
        groups,  # Number of groups
        BLOCK_SIZE: tl.constexpr,
    ):
        # Compute the 3D index of the output element
        pid_depth = tl.program_id(0)
        pid_height = tl.program_id(1)
        pid_width = tl.program_id(2)

        # Compute the offset in the output tensor
        out_depth = pid_depth
        out_height = pid_height
        out_width = pid_width

        # Compute the start and end indices in the input tensor
        in_depth_start = out_depth * stride_depth - padding_depth
        in_depth_end = in_depth_start + kernel_depth
        in_height_start = out_height * stride_height - padding_height
        in_height_end = in_height_start + kernel_height
        in_width_start = out_width * stride_width - padding_width
        in_width_end = in_width_start + kernel_width

        # Iterate over all batches
        for batch in range(batch_size):
            # Iterate over all groups
            for g in range(groups):
                # Compute the input and output channels for this group
                in_c = g * (in_channels // groups)
                out_c = g * (out_channels // groups)

                # Compute the offset in the input and output tensors
                in_offset = batch * in_channels * depth * height * width + in_c * depth * height * width + in_depth_start * height * width + in_height_start * width + in_width_start
                out_offset = batch * out_channels * depth_out * height_out * width_out + out_c * depth_out * height_out * width_out + out_depth * height_out * width_out + out_height * width_out + out_width

                # Load the weight
                weight = tl.load(weight_ptr + g * (out_channels // groups) * (in_channels // groups) * kernel_depth * kernel_height * kernel_width + out_c * (in_channels // groups) * kernel_depth * kernel_height * kernel_width + in_c * kernel_depth * kernel_height * kernel_width, mask=tl.arange(0, kernel_depth)[:, None, None] < kernel_depth, other=0.0)

                # Compute the input values
                x = tl.load(x_ptr + in_offset + tl.arange(0, kernel_depth)[:, None, None] * height * width + tl.arange(0, kernel_height)[None, :, None] * width + tl.arange(0, kernel_width)[None, None, :], mask=tl.arange(0, kernel_depth)[:, None, None] < kernel_depth, other=0.0)

                # Perform the convolution
                out = tl.sum(x * weight, axis=(0, 1, 2))

                # Add bias if present
                if bias_ptr is not None:
                    out += tl.load(bias_ptr + out_c, other=0.0)

                # Store the result
                tl.store(output_ptr + out_offset, out, mask=tl.arange(0, kernel_depth)[:, None, None] < kernel_depth)

    def _triton_conv_transpose3d(self, x: torch.Tensor, output: torch.Tensor):
        # Ensure input and output are on GPU
        assert x.is_cuda and output.is_cuda, "Tensors must be on CUDA."

        # Prepare input and output
        x = x.contiguous()
        output = output.contiguous()

        # Determine block size
        BLOCK_SIZE = 128

        # Compute the grid size
        grid = (self.depth_out, self.height_out, self.width_out)

        # Launch the kernel
        self._triton_conv_transpose3d_kernel[grid](x, self.weight, self.bias if self.bias else None, output, self.batch_size, self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], self.stride[0], self.stride[1], self.stride[2], self.padding[0], self.padding[1], self.padding[2], self.output_padding[0], self.output_padding[1], self.output_padding[2], self.groups, BLOCK_SIZE=BLOCK_SIZE)