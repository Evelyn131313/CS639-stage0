import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to weight tensor
    y_ptr,  # Pointer to output tensor
    x_shape,  # Shape of the input tensor (batch_size, in_channels, width, height, depth)
    w_shape,  # Shape of the weight tensor (out_channels, in_channels, kernel_width, kernel_height, kernel_depth)
    y_shape,  # Shape of the output tensor (batch_size, out_channels, width_out, height_out, depth_out)
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    dilation,  # Spacing between kernel elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    out_width_id = tl.program_id(2)
    out_height_id = tl.program_id(3)
    out_depth_id = tl.program_id(4)

    # Calculate the indices of the output element
    out_width = out_width_id * stride + padding
    out_height = out_height_id * stride + padding
    out_depth = out_depth_id * stride + padding

    # Calculate the indices of the input element
    in_width_start = out_width - dilation * (w_shape[2] - 1)
    in_height_start = out_height - dilation * (w_shape[3] - 1)
    in_depth_start = out_depth - dilation * (w_shape[4] - 1)

    # Mask to ensure we don't go out of bounds
    in_width_mask = (in_width_start + tl.arange(0, BLOCK_SIZE) < x_shape[2])
    in_height_mask = (in_height_start + tl.arange(0, BLOCK_SIZE) < x_shape[3])
    in_depth_mask = (in_depth_start + tl.arange(0, BLOCK_SIZE) < x_shape[4])

    # Initialize output value
    out = 0.0

    # Load input and weight values
    for k in range(w_shape[2]):
        for l in range(w_shape[3]):
            for m in range(w_shape[4]):
                in_width = in_width_start + k * dilation
                in_height = in_height_start + l * dilation
                in_depth = in_depth_start + m * dilation
                in_channel_id = tl.arange(0, w_shape[1])
                in_value = tl.load(x_ptr + ((batch_id * x_shape[1] + in_channel_id) * x_shape[2] + in_width_start + k * dilation) * x_shape[3] + in_height_start + l * dilation) * tl.load(w_ptr + ((out_channel_id * w_shape[1] + in_channel_id) * w_shape[2] + k) * w_shape[3] + l) * tl.load(w_ptr + ((out_channel_id * w_shape[1] + in_channel_id) * w_shape[2] + k) * w_shape[3] + l)
                out += in_value

    # Store the result
    tl.store(y_ptr + ((batch_id * y_shape[1] + out_channel_id) * y_shape[2] + out_width_id) * y_shape[3] + out_height_id, out)


def triton_conv3d(x: torch.Tensor, w: torch.Tensor):
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
    out_shape = (x.shape[0], w.shape[0], (x.shape[2] + 2 * w.shape[2] - 1 - 2 * w.shape[2]) // w.shape[2] + 1, (x.shape[3] + 2 * w.shape[3] - 1 - 2 * w.shape[3]) // w.shape[3] + 1, (x.shape[4] + 2 * w.shape[4] - 1 - 2 * w.shape[4]) // w.shape[4] + 1)
    out = torch.empty(out_shape, dtype=torch.float32, device=x.device)

    # Number of elements in the input and weight tensors
    n_elements_x = x.numel()
    n_elements_w = w.numel()
    n_elements_out = out.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: (
        x.shape[0],
        w.shape[0],
        ((out_shape[2] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]),
        ((out_shape[3] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]),
        ((out_shape[4] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]),
    )

    # Launch the Triton kernel
    conv3d_kernel[grid](x, w, out, x.shape, w.shape, out_shape, 1, 0, 1, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
        """
        return triton_conv3d(x, self.conv3d.weight)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel
width = 64
height = 64
depth = 64

def get_inputs():
    x = torch.rand(batch_size, in_channels, width, height, depth)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization