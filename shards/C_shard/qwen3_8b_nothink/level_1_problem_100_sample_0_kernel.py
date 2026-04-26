import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    predictions_ptr,  # Pointer to predictions
    targets_ptr,      # Pointer to targets
    out_ptr,          # Pointer to output
    n_elements,       # Total number of elements
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
    # Compute the product
    product = predictions * targets
    # Compute 1 - product
    one_minus_product = tl.where(mask, tl.where(product > 1, 0.0, 1.0 - product), 0.0)
    # Clamp to min=0
    clamped = tl.where(mask, tl.clamp(one_minus_product, 0.0, float('inf')), 0.0)
    # Sum the clamped values
    tl.store(out_ptr + offsets, clamped, mask=mask)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
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
    out = torch.empty((predictions.size(0),), dtype=torch.float32, device=predictions.device)

    # Number of elements in the tensor
    n_elements = predictions.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    hinge_loss_kernel[grid](predictions, targets, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return torch.mean(out)


class ModelNew(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use the Triton-based hinge loss computation
        return triton_hinge_loss(predictions, targets)