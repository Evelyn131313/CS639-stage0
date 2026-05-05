import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to weight tensor
    out_ptr,  # Pointer to output tensor
    n_channels,  # Number of input channels
    out_channels,  # Number of output channels
    kernel_size,  # Size of the convolution kernel
    stride,  # Stride of the convolution
    padding,  # Padding of the convolution
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    # Perform the convolution
    out = tl.zeros_like(x)
    for i in range(kernel_size):
        for j in range(kernel_size):
            out += x[:, :, padding+i:padding+i+1, padding+j:padding+j+1] * w[:, :, i, j]
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv2d(x: torch.Tensor, w: torch.Tensor):
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
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](x, w, out, n_channels, out_channels, kernel_size, stride, padding, n_elements, BLOCK_SIZE=BLOCK_SIZE, GROUPS=1)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        # Instead of "return self.conv1(x)", call our Triton-based convolution
        return triton_conv2d(x, self.conv1.weight)