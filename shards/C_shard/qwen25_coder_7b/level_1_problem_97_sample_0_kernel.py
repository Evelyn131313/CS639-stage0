import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def scaled_dot_product_attention_kernel(
    Q_ptr,  # Pointer to Q tensor
    K_ptr,  # Pointer to K tensor
    V_ptr,  # Pointer to V tensor
    out_ptr,  # Pointer to output tensor
    batch_size,  # Batch size
    num_heads,  # Number of heads
    sequence_length,  # Sequence length
    embedding_dimension,  # Embedding dimension
    scale_factor,  # Scale factor for dot product
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    row = tl.program_id(0)
    col = tl.program_id(1)
    head = tl.program_id(2)

    # Calculate the block offsets
    row_start = row * BLOCK_SIZE
    col_start = col * BLOCK_SIZE

    # Load Q, K, V values
    q = tl.load(Q_ptr + row_start * embedding_dimension + head * sequence_length * embedding_dimension, mask=tl.arange(BLOCK_SIZE) < sequence_length, other=0.0)
    k = tl.load(K_ptr + col_start * embedding_dimension + head * sequence_length * embedding_dimension, mask=tl.arange(BLOCK_SIZE) < sequence_length, other=0.0)
    v = tl.load(V_ptr + col_start * embedding_dimension + head * sequence_length * embedding_dimension, mask=tl.arange(BLOCK_SIZE) < sequence_length, other=0.0)

    # Perform the scaled dot product
    dot_product = tl.dot(q, k, allow_tf32=True)
    dot_product /= scale_factor

    # Softmax on the dot product
    max_val = tl.max(dot_product, axis=1, keepdim=True)
    exp_dot_product = tl.exp(dot_product - max_val)
    sum_exp_dot_product = tl.sum(exp_dot_product, axis=1, keepdim=True)
    softmax = exp_dot_product / sum_exp_dot_product

    # Store the result
    tl.store(out_ptr + row_start * embedding_dimension + col_start * embedding_dimension + head * sequence_length * embedding_dimension, softmax * v, mask=tl.arange(BLOCK_SIZE) < sequence_length)

def triton_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale_factor: float):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Tensors must be on CUDA."
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # Prepare output tensor
    out = torch.empty_like(Q)

    # Number of elements in the tensor
    batch_size = Q.shape[0]
    num_heads = Q.shape[1]
    sequence_length = Q.shape[2]
    embedding_dimension = Q.shape[3]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: (
        (batch_size * num_heads + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (sequence_length + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        num_heads
    )

    # Launch the Triton kernel
    scaled_dot_product_attention_kernel[grid](Q, K, V, out, batch_size, num_heads, sequence_length, embedding_dimension, scale_factor, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Scale factor for dot product
        scale_factor = 1.0 / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32, device=Q.device))
        # Instead of "return torch.nn.functional.scaled_dot_product_attention(Q, K, V)", call our Triton-based scaled dot product attention
        return triton_scaled_dot_product_attention(Q, K, V, scale_factor)