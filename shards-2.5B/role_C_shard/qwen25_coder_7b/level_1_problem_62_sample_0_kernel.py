import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to weight tensor
    y_ptr,  # Pointer to output tensor
    x_shape,  # Shape of input tensor (batch_size, in_channels, height, width)
    w_shape,  # Shape of weight tensor (out_channels, in_channels, kernel_height, kernel_width)
    y_shape,  # Shape of output tensor (batch_size, out_channels, height_out, width_out)
    stride_h,  # Stride height
    stride_w,  # Stride width
    padding_h,  # Padding height
    padding_w,  # Padding width
    kernel_height,  # Kernel height
    kernel_width,  # Kernel width
    BLOCK_SIZE_X: tl.constexpr,  # Block size in x dimension
    BLOCK_SIZE_Y: tl.constexpr,  # Block size in y dimension
    BLOCK_SIZE_C: tl.constexpr,  # Block size in channel dimension
):
    # Extract program indices
    n = tl.program_id(0)
    c = tl.program_id(1)
    y = tl.program_id(2)
    x = tl.program_id(3)

    # Compute the indices for the input tensor
    batch = n
    in_c = c
    in_y = y * stride_h - padding_h + y
    in_x = x * stride_w - padding_w + x

    # Compute the indices for the weight tensor
    out_c = c
    k_y = y
    k_x = x

    # Compute the indices for the output tensor
    out_y = y
    out_x = x

    # Load input and weight values with proper padding handling
    x_val = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)
    w_val = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)
    if in_y >= 0 and in_y < x_shape[2] and in_x >= 0 and in_x < x_shape[3]:
        x_val = tl.load(x_ptr + (batch * x_shape[1] * x_shape[2] * x_shape[3] + in_c * x_shape[2] * x_shape[3] + in_y * x_shape[3] + in_x) * 4, mask=True, other=0.0)
    if k_y >= 0 and k_y < kernel_height and k_x >= 0 and k_x < kernel_width:
        w_val = tl.load(w_ptr + (out_c * w_shape[1] * kernel_height * kernel_width + in_c * kernel_height * kernel_width + k_y * kernel_width + k_x) * 4, mask=True, other=0.0)

    # Perform the convolution
    out_val = tl.dot(x_val, w_val, allow_tf32=True)

    # Store the result
    if out_y >= 0 and out_y < y_shape[2] and out_x >= 0 and out_x < y_shape[3]:
        tl.store(y_ptr + (n * y_shape[1] * y_shape[2] * y_shape[3] + out_c * y_shape[2] * y_shape[3] + out_y * y_shape[3] + out_x) * 4, out_val)


def triton_conv2d(x: torch.Tensor, w: torch.Tensor, stride: int = 1, padding: int = 0):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    # Prepare output tensor
    out_channels, in_channels, kernel_height, kernel_width = w.shape
    batch_size, _, height, width = x.shape
    height_out = (height + 2 * padding - kernel_height) // stride + 1
    width_out = (width + 2 * padding - kernel_width) // stride + 1
    out = torch.empty((batch_size, out_channels, height_out, width_out), device=x.device, dtype=x.dtype)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE_X = 16
    BLOCK_SIZE_Y = 16
    BLOCK_SIZE_C = 8

    # Determine the number of blocks needed
    grid = (
        lambda meta: (
            (height_out + meta["BLOCK_SIZE_Y"] - 1) // meta["BLOCK_SIZE_Y"],
            (width_out + meta["BLOCK_SIZE_X"] - 1) // meta["BLOCK_SIZE_X"],
            out_channels,
            (batch_size + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"],
        ),
    )

    # Launch the Triton kernel
    conv2d_kernel[grid](x, w, out, x.shape, w.shape, out.shape, stride, stride, padding, padding, kernel_height, kernel_width, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_C)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(x, self.conv2d.weight, self.conv2d.stride[0], self.conv2d.padding[0])