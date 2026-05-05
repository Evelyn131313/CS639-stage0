import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to weight tensor
    y_ptr,  # Pointer to output tensor
    x_shape,  # Shape of the input tensor (batch_size, in_channels, depth, height, width)
    w_shape,  # Shape of the weight tensor (out_channels, in_channels, kernel_size_d, kernel_size_h, kernel_size_w)
    y_shape,  # Shape of the output tensor (batch_size, out_channels, depth_out, height_out, width_out)
    stride,  # Stride of the convolution (stride_d, stride_h, stride_w)
    padding,  # Padding applied to the input (padding_d, padding_h, padding_w)
    dilation,  # Spacing between kernel elements (dilation_d, dilation_h, dilation_w)
    BLOCK_SIZE_X: tl.constexpr,  # Block size in the x dimension
    BLOCK_SIZE_Y: tl.constexpr,  # Block size in the y dimension
    BLOCK_SIZE_Z: tl.constexpr,  # Block size in the z dimension
):
    # Extract block indices
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    out_d_id = tl.program_id(2)
    out_h_id = tl.program_id(3)
    out_w_id = tl.program_id(4)

    # Calculate the start and end indices for the block
    out_d_start = out_d_id * stride[0] - padding[0]
    out_h_start = out_h_id * stride[1] - padding[1]
    out_w_start = out_w_id * stride[2] - padding[2]

    # Iterate over the input and weight tensors
    for in_ch_id in range(w_shape[1]):
        for k_d in range(w_shape[2]):
            for k_h in range(w_shape[3]):
                for k_w in range(w_shape[4]):
                    # Calculate the input indices
                    in_d = out_d_start + k_d * dilation[0]
                    in_h = out_h_start + k_h * dilation[1]
                    in_w = out_w_start + k_w * dilation[2]

                    # Check if the input indices are within bounds
                    if (in_d >= 0 and in_d < x_shape[2] and
                        in_h >= 0 and in_h < x_shape[3] and
                        in_w >= 0 and in_w < x_shape[4]):
                        # Load the input and weight values
                        x_val = tl.load(x_ptr + (batch_id * x_shape[1] * x_shape[2] * x_shape[3] * x_shape[4] +
                                                in_ch_id * x_shape[2] * x_shape[3] * x_shape[4] +
                                                in_d * x_shape[3] * x_shape[4] +
                                                in_h * x_shape[4] +
                                                in_w) * 4, mask=True, other=0.0)
                        w_val = tl.load(w_ptr + (out_ch_id * w_shape[1] * w_shape[2] * w_shape[3] * w_shape[4] +
                                                in_ch_id * w_shape[2] * w_shape[3] * w_shape[4] +
                                                k_d * w_shape[3] * w_shape[4] +
                                                k_h * w_shape[4] +
                                                k_w) * 4, mask=True, other=0.0)

                        # Accumulate the result
                        y_val = tl.load(y_ptr + (batch_id * y_shape[1] * y_shape[2] * y_shape[3] * y_shape[4] +
                                                out_ch_id * y_shape[2] * y_shape[3] * y_shape[4] +
                                                out_d_id * y_shape[3] * y_shape[4] +
                                                out_h_id * y_shape[4] +
                                                out_w_id) * 4, mask=True, other=0.0)
                        y_val += x_val * w_val
                        tl.store(y_ptr + (batch_id * y_shape[1] * y_shape[2] * y_shape[3] * y_shape[4] +
                                          out_ch_id * y_shape[2] * y_shape[3] * y_shape[4] +
                                          out_d_id * y_shape[3] * y_shape[4] +
                                          out_h_id * y_shape[4] +
                                          out_w_id) * 4, y_val, mask=True)


def triton_conv3d(x: torch.Tensor, w: torch.Tensor, stride: tuple, padding: tuple, dilation: tuple):
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
    out_shape = (x.shape[0], w.shape[0], (x.shape[2] + 2 * padding[0] - dilation[0] * (w.shape[2] - 1) - 1) // stride[0] + 1,
                 (x.shape[3] + 2 * padding[1] - dilation[1] * (w.shape[3] - 1) - 1) // stride[1] + 1,
                 (x.shape[4] + 2 * padding[2] - dilation[2] * (w.shape[4] - 1) - 1) // stride[2] + 1)
    out = torch.empty(out_shape, dtype=torch.float32, device=x.device)

    # Number of elements in the output tensor
    n_elements = out.numel()
    BLOCK_SIZE_X = 32  # Tunable parameter for block size in the x dimension
    BLOCK_SIZE_Y = 32  # Tunable parameter for block size in the y dimension
    BLOCK_SIZE_Z = 4   # Tunable parameter for block size in the z dimension

    # Determine the number of blocks needed
    grid = lambda meta: ((out.shape[0], out.shape[1], out.shape[2], out.shape[3], out.shape[4]))

    # Launch the Triton kernel
    conv3d_kernel[grid](x, w, out, x.shape, w.shape, out.shape, stride, padding, dilation, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z)
    return out


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes using Triton kernels.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel in the form (kernel_size_d, kernel_size_h, kernel_size_w).
        stride (tuple, optional): Stride of the convolution in the form (stride_d, stride_h, stride_w). Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input in the form (padding_d, padding_h, padding_w). Defaults to (0, 0, 0).
        dilation (tuple, optional): Spacing between kernel elements in the form (dilation_d, dilation_h, dilation_w). Defaults to (1, 1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size, dtype=torch.float32, device='cuda'))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = nn.Parameter(torch.randn(out_channels, dtype=torch.float32, device='cuda')) if bias else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return triton_conv3d(x, self.weight, self.stride, self.padding, self.dilation)