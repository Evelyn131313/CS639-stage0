import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scaled_dot_product_attention_kernel(
    Q_ptr, K_ptr, V_ptr, out_ptr,
    batch_size, num_heads, sequence_length, embedding_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the index for the current thread
    pid = tl.program_id(0)
    # Compute the offset for the current thread
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the index in the tensor
    idx = offset

    # Check if the index is within bounds
    mask = idx < sequence_length * embedding_dim

    # Load Q, K, V
    Q = tl.load(Q_ptr + idx, mask=mask, other=0.0)
    K = tl.load(K_ptr + idx, mask=mask, other=0.0)
    V = tl.load(V_ptr + idx, mask=mask, other=0.0)

    # Compute Q*K^T
    attn_weights = tl.dot(Q, K)
    # Apply softmax
    attn_weights = tl.softmax(attn_weights, axis=-1)
    # Multiply with V
    out = tl.dot(attn_weights, V)

    # Store the result
    tl.store(out_ptr + idx, out, mask=mask)


def triton_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    """
    Custom Triton implementation of scaled dot product attention.
    """
    # Ensure inputs are on GPU and contiguous
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Tensors must be on CUDA."
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # Prepare output tensor
    out = torch.empty_like(Q)

    # Parameters
    batch_size = Q.size(0)
    num_heads = Q.size(1)
    sequence_length = Q.size(2)
    embedding_dim = Q.size(3)

    # Determine block size
    BLOCK_SIZE = 128  # Tunable parameter

    # Compute grid size
    grid = (sequence_length * embedding_dim + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    scaled_dot_product_attention_kernel[grid](
        Q, K, V, out,
        batch_size, num_heads, sequence_length, embedding_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return triton_scaled_dot_product_attention(Q, K, V)