import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    predictions_ptr,  # Pointer to predictions tensor
    targets_ptr,     # Pointer to targets tensor
    out_ptr,         # Pointer to output tensor
    n_predictions,   # Total number of predictions
    n_classes,       # Total number of classes
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_predictions

    # Load input values
    predictions = tl.load(predictions_ptr + offsets * n_classes, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0)

    # Compute log probabilities
    log_probs = tl.log(predictions)
    # Gather the log probabilities for the correct classes
    gather_indices = targets * n_classes + offsets
    log_probs_correct = tl.load(predictions_ptr + gather_indices, mask=mask, other=0.0)

    # Compute the negative log likelihood
    loss = -log_probs_correct
    # Sum the losses and average them
    tl.store(out_ptr + offsets, loss, mask=mask)


def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor):
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
    out = torch.empty((predictions.shape[0],), dtype=torch.float32, device=predictions.device)

    # Number of predictions and classes
    n_predictions = predictions.shape[0]
    n_classes = predictions.shape[1]
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_predictions + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    cross_entropy_kernel[grid](predictions, targets, out, n_predictions, n_classes, BLOCK_SIZE=BLOCK_SIZE)
    return out.mean()


class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks using Triton kernels.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)