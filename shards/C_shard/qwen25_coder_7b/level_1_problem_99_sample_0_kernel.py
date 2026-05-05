import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    margin,
    out_ptr,
    n_triplets,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_triplets
    anchor = tl.load(anchor_ptr + offsets * dim, mask=mask, other=0.0)
    positive = tl.load(positive_ptr + offsets * dim, mask=mask, other=0.0)
    negative = tl.load(negative_ptr + offsets * dim, mask=mask, other=0.0)
    
    diff_pos = anchor - positive
    diff_neg = anchor - negative
    pos_dist = tl.math.sqrt(tl.math.sum(diff_pos * diff_pos, axis=0))
    neg_dist = tl.math.sqrt(tl.math.sum(diff_neg * diff_neg, axis=0))
    
    margin = tl.convert_element_type(margin, pos_dist.dtype)
    loss = tl.maximum(margin + neg_dist - pos_dist, 0.0)
    
    tl.store(out_ptr + offsets, loss, mask=mask)


def triton_triplet_margin_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float):
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda, "Tensors must be on CUDA."
    anchor = anchor.contiguous()
    positive = positive.contiguous()
    negative = negative.contiguous()

    out = torch.empty_like(anchor, dtype=torch.float32)

    n_triplets = anchor.numel() // dim
    BLOCK_SIZE = 256

    grid = lambda meta: ((n_triplets + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    triplet_margin_loss_kernel[grid](anchor, positive, negative, margin, out, n_triplets, BLOCK_SIZE=BLOCK_SIZE)
    return out.sum()


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)