import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batch_norm_2d_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    bias_ptr,  # Pointer to bias tensor
    mean_ptr,  # Pointer to mean tensor
    inv_std_ptr,  # Pointer to inverse standard deviation tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements in the input tensor
    channels,  # Number of channels in the input tensor
    height,  # Height of the input tensor
    width,  # Width of the input tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + tl.arange(channels), mask=mask, other=1.0)
    bias = tl.load(bias_ptr + tl.arange(channels), mask=mask, other=0.0)
    mean = tl.load(mean_ptr + tl.arange(channels), mask=mask, other=0.0)
    inv_std = tl.load(inv_std_ptr + tl.arange(channels), mask=mask, other=1.0)

    # Normalize the input
    x_norm = (x - mean) * inv_std
    out = x_norm * weight + bias

    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_batch_norm_2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, mean: torch.Tensor, inv_std: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    mean = mean.contiguous()
    inv_std = inv_std.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    batch_norm_2d_kernel[grid](x, weight, bias, mean, inv_std, out, n_elements, channels, height, width, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.bn.running_mean.to(x.device)
        inv_std = 1 / torch.sqrt(self.bn.running_var.to(x.device) + self.bn.eps)
        weight = self.bn.weight.to(x.device)
        bias = self.bn.bias.to(x.device)
        return triton_batch_norm_2d(x, weight, bias, mean, inv_std)