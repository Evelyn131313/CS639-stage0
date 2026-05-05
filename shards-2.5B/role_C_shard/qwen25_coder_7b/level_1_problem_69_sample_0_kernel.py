import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    n_out_channels,  # Number of output channels
    kernel_size_h,  # Height of the kernel
    kernel_size_w,  # Width of the kernel
    stride_h,  # Stride in height
    stride_w,  # Stride in width
    padding_h,  # Padding in height
    padding_w,  # Padding in width
    input_shape_h,  # Height of the input
    input_shape_w,  # Width of the input
    output_shape_h,  # Height of the output
    output_shape_w,  # Width of the output
    BLOCK_SIZE_X: tl.constexpr,  # Block size in X dimension
    BLOCK_SIZE_Y: tl.constexpr,  # Block size in Y dimension
    BLOCK_SIZE_Z: tl.constexpr,  # Block size in Z dimension
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    block_start_x = pid_x * BLOCK_SIZE_X
    block_start_y = pid_y * BLOCK_SIZE_Y
    block_start_z = pid_z * BLOCK_SIZE_Z

    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE_X)
    offsets_y = block_start_y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_z = block_start_z + tl.arange(0, BLOCK_SIZE_Z)

    mask_x = offsets_x < output_shape_w
    mask_y = offsets_y < output_shape_h
    mask_z = offsets_z < n_out_channels

    out_idx = (offsets_z * output_shape_h + offsets_y) * output_shape_w + offsets_x
    tl.store(out_ptr + out_idx, 0.0, mask=mask_z & mask_y & mask_x)

    for ky in range(kernel_size_h):
        ky_idx = block_start_y + tl.arange(0, BLOCK_SIZE_Y)
        ky_mask = ky_idx < kernel_size_h

        for kx in range(kernel_size_w):
            kx_idx = block_start_x + tl.arange(0, BLOCK_SIZE_X)
            kx_mask = kx_idx < kernel_size_w

            ky_masked = ky_idx < kernel_size_h
            kx_masked = kx_idx < kernel_size_w

            weight_idx = (ky_idx * kernel_size_w + kx_idx) * n_out_channels
            weight_mask = ky_masked & kx_masked

            for ic in range(n_out_channels):
                ic_idx = block_start_z + tl.arange(0, BLOCK_SIZE_Z)
                ic_mask = ic_idx < n_out_channels

                ic_masked = ic_idx < n_out_channels

                input_idx = ((ic_idx + padding_z) * input_shape_h + (ky_idx + padding_y)) * input_shape_w + (kx_idx + padding_x)
                input_mask = ic_masked & ky_masked & kx_masked

                weight_value = tl.load(weight_ptr + weight_idx + ic_idx, mask=weight_mask, other=0.0)
                input_value = tl.load(x_ptr + input_idx, mask=input_mask, other=0.0)

                out_value = tl.load(out_ptr + out_idx, mask=mask_z & mask_y & mask_x, other=0.0)
                out_value += weight_value * input_value

                tl.store(out_ptr + out_idx, out_value, mask=mask_z & mask_y & mask_x)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()

    n_batch, n_in_channels, input_shape_h, input_shape_w = x.shape
    n_out_channels, kernel_size_h, kernel_size_w, n_in_channels = weight.shape
    stride_h, stride_w = 1, 1
    padding_h, padding_w = 0, 0
    output_shape_h = (input_shape_h - 1) * stride_h - 2 * padding_h + kernel_size_h
    output_shape_w = (input_shape_w - 1) * stride_w - 2 * padding_w + kernel_size_w

    out = torch.empty((n_batch, n_out_channels, output_shape_h, output_shape_w), dtype=x.dtype, device=x.device)

    n_elements = out.numel()
    BLOCK_SIZE_X = 32
    BLOCK_SIZE_Y = 32
    BLOCK_SIZE_Z = 4

    grid = lambda meta: (
        (output_shape_w + meta["BLOCK_SIZE_X"] - 1) // meta["BLOCK_SIZE_X"],
        (output_shape_h + meta["BLOCK_SIZE_Y"] - 1) // meta["BLOCK_SIZE_Y"],
        (n_out_channels + meta["BLOCK_SIZE_Z"] - 1) // meta["BLOCK_SIZE_Z"],
    )

    conv_transpose2d_kernel[grid](x, weight, out, n_elements, n_out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, input_shape_h, input_shape_w, output_shape_h, output_shape_w, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, kernel_size[0], kernel_size[1], in_channels))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = triton_conv_transpose2d(x, self.weight)
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out