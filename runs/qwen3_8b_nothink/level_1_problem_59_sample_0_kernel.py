import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch, in_channels, depth, height, width)
    kernel_size,  # (kz, ky, kx)
    stride,  # (sz, sy, sx)
    padding,  # (pz, py, px)
    dilation,  # (dz, dy, dx)
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # Get the batch, in_channels, out_channels, depth, height, width
    batch, in_channels, out_channels, depth, height, width = input_shape
    kz, ky, kx = kernel_size
    sz, sy, sx = stride
    pz, py, px = padding
    dz, dy, dx = dilation

    # Compute the output dimensions
    out_depth = (depth + 2 * pz - dz * (dilation[0] - 1) - 1) // sz + 1
    out_height = (height + 2 * py - dy * (dilation[1] - 1) - 1) // sy + 1
    out_width = (width + 2 * px - dx * (dilation[2] - 1) - 1) // sx + 1

    # Compute the output index
    # Each thread handles one output element
    out_idx = tl.program_id(0)
    out_batch = out_idx // (out_channels * out_depth * out_height * out_width)
    out_channel = (out_idx // (out_depth * out_height * out_width)) % out_channels
    out_depth_idx = (out_idx // (out_height * out_width)) % out_depth
    out_height_idx = (out_idx // out_width) % out_height
    out_width_idx = out_idx % out_width

    # Compute the input indices for each spatial position
    # For each input channel in the group
    for group in range(GROUPS):
        in_channel = (in_channels // GROUPS) * group + (out_channel // (out_channels // GROUPS)) % (in_channels // GROUPS)
        # Load the weights for this group and channel
        weight_offset = group * in_channels * out_channels * kz * ky * kx
        weight_offset += out_channel * in_channels * kz * ky * kx
        weight_offset += in_channel * kz * ky * kx
        weight = tl.load(weight_ptr + weight_offset + tl.arange(0, kz * ky * kx), other=0.0)

        # Compute the input indices for this output position
        in_depth_start = out_depth_idx * sz - pz + dz * (dilation[0] - 1)
        in_depth_start = tl.max(tl.min(in_depth_start, depth - 1), 0)
        in_depth_end = in_depth_start + kz
        in_height_start = out_height_idx * sy - py + dy * (dilation[1] - 1)
        in_height_start = tl.max(tl.min(in_height_start, height - 1), 0)
        in_height_end = in_height_start + ky
        in_width_start = out_width_idx * sx - px + dx * (dilation[2] - 1)
        in_width_start = tl.max(tl.min(in_width_start, width - 1), 0)
        in_width_end = in_width_start + kx

        # Iterate over the kernel
        for dz_idx in range(kz):
            for dy_idx in range(ky):
                for dx_idx in range(kx):
                    in_depth = in_depth_start + dz_idx * dilation[0]
                    in_height = in_height_start + dy_idx * dilation[1]
                    in_width = in_width_start + dx_idx * dilation[2]
                    in_idx = (out_batch * in_channels * depth * height * width) + (in_channel * depth * height * width) + (in_depth * height * width) + (in_height * width) + in_width
                    input_val = tl.load(input_ptr + in_idx, other=0.0)
                    output_val = tl.load(output_ptr + out_idx, other=0.0)
                    output_val += input_val * weight[dz_idx * ky * kx + dy_idx * kx + dx_idx]
                    tl.store(output_ptr + out_idx, output_val)

    return


def triton_conv3d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: tuple, padding: tuple, dilation: tuple):
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

    # Compute output shape
    batch, in_channels, depth, height, width = input.shape
    out_channels = weight.shape[1]
    kz, ky, kx = weight.shape[2:]
    sz, sy, sx = stride
    pz, py, px = padding
    dz, dy, dx = dilation

    out_depth = (depth + 2 * pz - dz * (dilation[0] - 1) - 1) // sz + 1
    out_height = (height + 2 * py - dy * (dilation[1] - 1) - 1) // sy + 1
    out_width = (width + 2 * px - dx * (dilation[2] - 1) - 1) // sx + 1

    output = torch.empty((batch, out_channels, out_depth, out_height, out_width), dtype=input.dtype, device=input.device)

    # Prepare input shape and kernel size
    input_shape = (batch, in_channels, out_channels, depth, height, width)
    kernel_size = (kz, ky, kx)
    stride = (sz, sy, sx)
    padding = (pz, py, px)
    dilation = (dz, dy, dx)

    # Determine the number of blocks needed
    num_output_elements = batch * out_channels * out_depth * out_height * out_width
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((num_output_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv3d_kernel[grid](input, weight, output, input_shape, kernel_size, stride, padding, dilation, BLOCK_SIZE, weight.shape[0] // (in_channels // (out_channels // weight.shape[0])))

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out, depth_out).
        """
        batch, in_channels, depth, height, width = x.shape
        out_channels = self.out_channels
        kz = self.kernel_size
        ky = self.kernel_size
        kx = 1
        sz = self.stride
        sy = self.stride
        sx = self.stride
        pz = self.padding
        py = self.padding
        px = self.padding
        dz = self.dilation
        dy = self.dilation
        dx = self.dilation
        groups = self.groups
        bias = self.bias

        # Create weight tensor
        weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, kz, ky, kx, device=x.device, dtype=x.dtype))
        # Create bias tensor if needed
        if bias:
            bias = torch.nn.Parameter(torch.randn(out_channels, device=x.device, dtype=x.dtype))
        else:
            bias = None

        # Perform the convolution
        output = triton_conv3d(x, weight, bias, (sz, sy, sx), (pz, py, px), (dz, dy, dx))

        return output