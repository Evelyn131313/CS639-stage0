import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height_in: tl.constexpr,
    width_in: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Compute the index of the current thread
    pid = tl.program_id(0)
    # Compute the index of the current thread within the block
    block_id = pid // (BLOCK_SIZE * BLOCK_SIZE)
    block_idx = pid % (BLOCK_SIZE * BLOCK_SIZE)
    # Compute the row and column within the block
    row = block_idx // BLOCK_SIZE
    col = block_idx % BLOCK_SIZE

    # Compute the position in the output
    out_h = block_id // (width_in // stride)
    out_w = block_id % (width_in // stride)

    # Compute the corresponding input position
    in_h = out_h * stride - padding
    in_w = out_w * stride - padding

    # Compute the offset in the input tensor
    in_offset = (out_h * stride - padding) * width_in + (out_w * stride - padding)
    in_offset = in_offset * in_channels + pid % in_channels

    # Load input data
    input_val = tl.load(input_ptr + in_offset, other=0.0)

    # Compute the weight index
    weight_offset = (pid % in_channels) * out_channels + pid // in_channels
    weight_val = tl.load(weight_ptr + weight_offset, other=0.0)

    # Compute the output value
    output_val = input_val * weight_val

    # Store the output value
    output_ptr_idx = (out_h * width_in // stride) + out_w
    output_ptr_idx = output_ptr_idx * out_channels + pid % out_channels
    tl.store(output_ptr + output_ptr_idx, output_val)


def triton_depthwise_conv(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, batch_size: int, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
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
    output = torch.zeros(batch_size, out_channels, (input.size(2) + 2 * padding) // stride, (input.size(3) + 2 * padding) // stride, device=input.device)

    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size
    GROUP_SIZE = in_channels  # Each group processes one input channel

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    depthwise_conv_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, input.size(2), input.size(3), kernel_size, stride, padding, BLOCK_SIZE, GROUP_SIZE)
    return output


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel using a custom Triton kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, device='cuda'))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, device='cuda'))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        batch_size = x.size(0)
        height_in = x.size(2)
        width_in = x.size(3)

        # Apply the custom Triton kernel
        output = triton_depthwise_conv(x, self.weight, self.bias, batch_size, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

        return output