import torch
import torch.nn as nn
import triton
import triton.language as tl


BLOCK_SIZE = 128  # Tunable parameter for block size


@triton.jit
def triplet_loss_kernel(
    anchor_ptr,  # Pointer to anchor tensor
    positive_ptr,  # Pointer to positive tensor
    negative_ptr,  # Pointer to negative tensor
    out_ptr,  # Pointer to output tensor (block sums)
    batch_size,  # Total number of triplets
    margin,  # Margin parameter
    BLOCK_SIZE: tl.constexpr
):
    # Each block processes a contiguous range of triplets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    # Load the anchor, positive, and negative values for the current block
    a = tl.load(anchor_ptr + offsets, mask=mask, other=0.0)
    p = tl.load(positive_ptr + offsets, mask=mask, other=0.0)
    n = tl.load(negative_ptr + offsets, mask=mask, other=0.0)

    # Compute the loss for each triplet in the block
    loss = margin + (a - p) ** 2 - (a - n) ** 2
    loss = tl.where(loss > 0, loss, 0.0)

    # Sum the losses in the block
    block_sum = tl.sum(loss, axis=0)

    # Store the block sum in the output tensor
    tl.store(out_ptr + pid, block_sum)


def triton_triplet_loss(anchor, positive, negative, margin, batch_size):
    """
    Compute the triplet loss using a Triton kernel.
    """
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Tensors must be on CUDA."
    # Ensure they are contiguous
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()

    # Determine the number of blocks needed
    num_blocks = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Output tensor to store block sums
    output = torch.empty(num_blocks, dtype=torch.float32, device='cuda')

    # Launch the kernel
    grid = (num_blocks,)
    triplet_loss_kernel[grid](anchor, positive, negative, output, batch_size, margin, BLOCK_SIZE=BLOCK_SIZE)

    # Sum all the block sums
    total_loss = torch.sum(output)

    # Compute the mean loss
    mean_loss = total_loss / batch_size

    return mean_loss


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_loss(anchor, positive, negative, self.margin, anchor.size(0))