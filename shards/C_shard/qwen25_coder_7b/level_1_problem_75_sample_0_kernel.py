import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    out_ptr,  # Pointer to output tensor
    stride_h, stride_w,  # Stride height and width
    padding_h, padding_w,  # Padding height and width
    dilation_h, dilation_w,  # Dilation height and width
    groups,  # Number of groups
    in_channels, out_channels, kernel_size_h, kernel_size_w,  # Dimensions of the input, output, and kernel
    batch_size, height_in, width_in,  # Dimensions of the input tensor
    height_out, width_out,  # Dimensions of the output tensor
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = (batch_size * height_out * width_out) // BLOCK_SIZE
    if pid >= grid_size:
        return

    # Calculate the output coordinates
    b = pid // (height_out * width_out)
    h = (pid % (height_out * width_out)) // width_out
    w = pid % width_out

    # Calculate the input coordinates
    h_in = h * stride_h - padding_h + dilation_h * (tl.arange(0, kernel_size_h) - kernel_size_h // 2)
    w_in = w * stride_w - padding_w + dilation_w * (tl.arange(0, kernel_size_w) - kernel_size_w // 2)

    # Calculate the index in the input tensor
    h_in = h_in + tl.arange(0, kernel_size_h)
    w_in = w_in + tl.arange(0, kernel_size_w)
    mask_h = (h_in >= 0) & (h_in < height_in)
    mask_w = (w_in >= 0) & (w_in < width_in)
    mask = mask_h & mask_w
    h_in = tl.where(mask, h_in, 0)
    w_in = tl.where(mask, w_in, 0)

    # Calculate the output index
    idx_out = (b * out_channels + tl.arange(0, out_channels)) * height_out * width_out + h * width_out + w

    # Calculate the weight index
    idx_weight = (tl.arange(0, out_channels) * groups // out_channels) * kernel_size_h * kernel_size_w + \
                 (tl.arange(0, kernel_size_h) * kernel_size_w + tl.arange(0, kernel_size_w))

    # Load input values and weights
    x = tl.load(x_ptr + (b * in_channels + tl.arange(0, in_channels)) * height_in * width_in + h_in * width_in + w_in, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + idx_weight, mask=mask, other=0.0)

    # Perform the convolution
    acc = tl.zeros((out_channels,), dtype=tl.float32)
    for i in range(kernel_size_h):
        for j in range(kernel_size_w):
            acc += x[i * width_in + j] * weight[i * kernel_size_w + j]

    # Store the result
    tl.store(out_ptr + idx_out, acc)

def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, stride: tuple, padding: tuple, dilation: tuple, groups: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()

    # Prepare output tensor
    out_channels, in_channels, kernel_size_h, kernel_size_w = weight.shape
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    batch_size, in_channels, height_in, width_in = x.shape
    height_out = (height_in - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_size_h - 1) + 1
    width_out = (width_in - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_size_w - 1) + 1
    out = torch.empty((batch_size, out_channels, height_out, width_out), dtype=torch.float32, device=x.device)

    # Number of elements in the output tensor
    n_elements = out.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose2d_kernel[grid](x, weight, out, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups, in_channels, out_channels, kernel_size_h, kernel_size_w, batch_size, height_in, width_in, height_out, width_out, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose2d.weight.data
        bias = self.conv_transpose2d.bias.data if self.conv_transpose2d.bias is not None else None
        return triton_conv_transpose2d(x, weight, self.conv_transpose2d.stride, self.conv_transpose2d.padding, self.conv_transpose2d.dilation, self.conv_transpose2d.groups)