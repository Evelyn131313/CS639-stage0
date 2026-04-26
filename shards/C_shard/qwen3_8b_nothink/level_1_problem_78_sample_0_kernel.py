import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Compute the 2D index in the output
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_z = tl.program_id(2)

    # Compute the offset in the output
    oh = pid_h * BLOCK_H
    ow = pid_w * BLOCK_W
    oz = pid_z

    # Compute the corresponding input position
    # Output shape: (batch, out_channels, H, W)
    # Input shape: (batch, in_channels, H_in, W_in)
    # Output shape after transposed conv: H_out = (H_in - 1) * stride + kernel_size - 2 * padding
    # For simplicity, assume H_out = H_in * stride + kernel_size - 2 * padding
    # We will compute the input positions for each output position

    # For each output position (oh, ow)
    # We loop over all possible input positions (ih, iw) that can contribute to (oh, ow)
    # The input positions are determined by the transposed convolution formula:
    # ih = (oh - (kernel_h - 1) * stride_h + 2 * padding_h) // stride_h
    # iw = (ow - (kernel_w - 1) * stride_w + 2 * padding_w) // stride_w

    # We loop over the output channels
    for oc in range(out_channels):
        # Loop over the output height
        for oh_idx in range(BLOCK_H):
            # Loop over the output width
            for ow_idx in range(BLOCK_W):
                # Compute the actual output position
                o_h = oh + oh_idx
                o_w = ow + ow_idx

                # Compute the input position
                i_h = (o_h - (kernel_h - 1) * stride_h + 2 * padding_h) // stride_h
                i_w = (o_w - (kernel_w - 1) * stride_w + 2 * padding_w) // stride_w

                # Compute the input and output indices
                i_idx = oz * in_channels * height * width + oc * height * width + i_h * width + i_w
                w_idx = oc * in_channels * kernel_h * kernel_w + (i_h % kernel_h) * kernel_w + (i_w % kernel_w)
                o_idx = oz * out_channels * height * width + oc * height * width + o_h * width + o_w

                # Load input and weight
                input_val = tl.load(input_ptr + i_idx, 0.0)
                weight_val = tl.load(weight_ptr + w_idx, 0.0)

                # Compute output
                output_val = input_val * weight_val

                # Store output
                tl.store(output_ptr + o_idx, output_val, 0.0)

def triton_conv_transpose2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    padding_h: int,
    padding_w: int,
):
    """
    This function wraps the Triton kernel call for transposed convolution.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()

    # Output shape
    output_h = (input.shape[2] - 1) * stride_h + kernel_h - 2 * padding_h
    output_w = (input.shape[3] - 1) * stride_w + kernel_w - 2 * padding_w
    output = torch.empty((batch_size, out_channels, output_h, output_w), dtype=input.dtype, device=input.device)

    # Kernel parameters
    BLOCK_H = 16
    BLOCK_W = 16

    # Grid size
    grid_h = (output_h + BLOCK_H - 1) // BLOCK_H
    grid_w = (output_w + BLOCK_W - 1) // BLOCK_W
    grid_z = batch_size

    # Launch the kernel
    conv_transpose2d_kernel[grid_z, grid_w, grid_h](
        input, weight, output,
        batch_size, in_channels, out_channels,
        kernel_h, kernel_w, stride_h, stride_w,
        padding_h, padding_w,
        BLOCK_H, BLOCK_W
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure the input is on the GPU
        x = x.contiguous()
        batch_size = x.shape[0]
        input_h = x.shape[2]
        input_w = x.shape[3]

        # Compute output dimensions
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        output_h = (input_h - 1) * stride_h + kernel_h - 2 * padding_h
        output_w = (input_w - 1) * stride_w + kernel_w - 2 * padding_w

        # Initialize output
        output = torch.empty((batch_size, self.out_channels, output_h, output_w), dtype=x.dtype, device=x.device)

        # Create weight tensor (random initialization)
        weight = torch.randn((self.out_channels, self.in_channels, kernel_h, kernel_w), dtype=x.dtype, device=x.device)

        # Call the Triton kernel
        output = triton_conv_transpose2d(x, weight, batch_size, self.in_channels, self.out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w)

        # Add bias if needed
        if self.bias:
            bias = torch.nn.Parameter(torch.randn(self.out_channels, device=x.device))
            output = output + bias.view(1, self.out_channels, 1, 1)

        return output