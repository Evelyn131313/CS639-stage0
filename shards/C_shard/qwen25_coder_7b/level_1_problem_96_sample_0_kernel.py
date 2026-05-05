import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def smooth_l1_loss_kernel(
    predictions_ptr,  # Pointer to predictions
    targets_ptr,  # Pointer to targets
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    delta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    # Compute Smooth L1 Loss
    diff = predictions - targets
    abs_diff = tl.abs(diff)
    out = tl.where(abs_diff < delta, 0.5 * diff * diff, abs_diff - 0.5 * delta)
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, delta: float = 1.0):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    # Prepare output tensor
    out = torch.empty_like(predictions)

    # Number of elements in the tensor
    n_elements = predictions.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    smooth_l1_loss_kernel[grid](predictions, targets, out, n_elements, delta=delta, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)