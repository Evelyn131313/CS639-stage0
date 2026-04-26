import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def transposed_conv3d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # Shape of input tensor (batch, in_channels, depth, height, width)
    weight_shape,  # Shape of weight tensor (out_channels, in_channels // groups, kernel_depth, kernel_height, kernel_width)
    output_shape,  # Shape of output tensor (batch, out_channels, depth_out, height_out, width_out)
    stride,  # Stride tuple (stride_depth, stride_height, stride_width)
    padding,  # Padding tuple (padding_depth, padding_height, padding_width)
    output_padding,  # Output padding tuple (output_padding_depth, output_padding_height, output_padding_width)
    groups,  # Number of groups
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread processes a single output element
    # Compute the output indices (d, h, w)
    # Get the batch index
    batch_idx = tl.program_id(0)
    # Get the output channel index
    out_ch_idx = tl.program_id(1)
    # Get the output depth index
    out_d = tl.program_id(2)
    # Get the output height index
    out_h = tl.program_id(3)
    # Get the output width index
    out_w = tl.program_id(4)

    # Calculate the corresponding input indices
    input_d = (out_d - output_padding[0]) // stride[0]
    input_h = (out_h - output_padding[1]) // stride[1]
    input_w = (out_w - output_padding[2]) // stride[2]

    # Calculate the input and output dimensions
    input_depth = input_shape[2]
    input_height = input_shape[3]
    input_width = input_shape[4]
    output_depth = output_shape[2]
    output_height = output_shape[3]
    output_width = output_shape[4]

    # Calculate the input channel index
    in_ch_idx = (out_ch_idx % (input_shape[1] // groups)) * groups

    # Initialize the output value
    out_val = tl.zeros((), dtype=tl.float32)

    # Iterate over the kernel dimensions
    for kd in range(weight_shape[2]):
        for kh in range(weight_shape[3]):
            for kw in range(weight_shape[4]):
                # Calculate the input indices for this kernel position
                input_d_k = input_d + kd
                input_h_k = input_h + kh
                input_w_k = input_w + kw

                # Check if the input indices are within bounds
                if (input_d_k >= 0 and input_d_k < input_depth and
                    input_h_k >= 0 and input_h_k < input_height and
                    input_w_k >= 0 and input_w_k < input_width):
                    # Load input value
                    input_val = tl.load(input_ptr + (batch_idx * input_shape[1] + in_ch_idx) + (input_d_k * input_height * input_width + input_h_k * input_width + input_w_k), dtype=tl.float32)
                    # Load weight value
                    weight_val = tl.load(weight_ptr + (out_ch_idx * (input_shape[1] // groups) * weight_shape[2] * weight_shape[3] * weight_shape[4] + in_ch_idx // groups * weight_shape[2] * weight_shape[3] * weight_shape[4] + kd * weight_shape[3] * weight_shape[4] + kh * weight_shape[4] + kw), dtype=tl.float32)
                    # Accumulate the result
                    out_val += input_val * weight_val

    # Store the result
    tl.store(output_ptr + (batch_idx * output_shape[1] + out_ch_idx) + (out_d * output_height * output_width + out_h * output_width + out_w), out_val, mask=out_val != 0)


def triton_transposed_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride, padding, output_padding, groups):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    output = torch.empty_like(x)

    # Calculate output shape
    batch_size = x.size(0)
    in_channels = x.size(1)
    out_channels = weight.size(0)
    kernel_size = (weight.size(2), weight.size(3), weight.size(4))
    depth = x.size(2)
    height = x.size(3)
    width = x.size(4)

    # Calculate output dimensions
    out_depth = (depth - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0]
    out_height = (height - 1) * stride[1] - 2 * padding[1] + kernel_size[1] + output_padding[1]
    out_width = (width - 1) * stride[2] - 2 * padding[2] + kernel_size[2] + output_padding[2]

    # Calculate the number of elements in the output
    n_elements = batch_size * out_channels * out_depth * out_height * out_width

    # Determine the number of blocks needed
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    transposed_conv3d_kernel[grid](x, weight, output, x.size(), weight.size(), output.size(), stride, padding, output_padding, groups, BLOCK_SIZE=BLOCK_SIZE)
    return output


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using a custom Triton kernel.
        """
        # Get the weight tensor
        weight = torch.randn(self.out_channels, self.in_channels // self.groups, *self.kernel_size, device=x.device, dtype=x.dtype)
        # Get the bias tensor
        bias = torch.randn(self.out_channels, device=x.device, dtype=x.dtype) if self.bias else None
        # Perform the transposed convolution
        return triton_transposed_conv3d(x, weight, bias, self.stride, self.padding, self.output_padding, self.groups)