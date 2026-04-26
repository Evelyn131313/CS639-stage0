import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_loss_kernel(
    anchor_ptr,  # Pointer to anchor tensor
    positive_ptr,  # Pointer to positive tensor
    negative_ptr,  # Pointer to negative tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in each tensor
    margin,  # Triplet margin
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements

    # Load anchor, positive, and negative values
    a = tl.load(anchor_ptr + offsets, mask=mask, other=0.0)
    p = tl.load(positive_ptr + offsets, mask=mask, other=0.0)
    n = tl.load(negative_ptr + offsets, mask=mask, other=0.0)

    # Compute distances
    dist_ap = a - p
    dist_an = a - n

    # Compute squared distances
    dist_ap = dist_ap * dist_ap
    dist_an = dist_an * dist_an

    # Compute triplet loss
    loss = tl.maximum(tl.sqrt(dist_ap) - tl.sqrt(dist_an) + margin, 0.0)

    # Store the result
    tl.store(output_ptr + offsets, loss, mask=mask)


def triton_triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float):
    """
    This function wraps the Triton kernel call for triplet loss computation.
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Tensors must be on CUDA."
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()

    # Prepare output tensor
    output = torch.empty(anchor.shape[0], dtype=torch.float32, device=anchor.device)

    # Number of elements in the tensor
    n_elements = anchor.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    triplet_loss_kernel[grid](anchor, positive, negative, output, n_elements, margin, BLOCK_SIZE=BLOCK_SIZE)
    return output.mean()


class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks using a custom Triton kernel.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute triplet loss using the custom Triton kernel
        loss = triton_triplet_loss(anchor, positive, negative, self.margin)
        return loss