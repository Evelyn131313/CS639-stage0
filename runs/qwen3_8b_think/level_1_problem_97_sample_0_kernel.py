import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.num_heads = 32
        self.sequence_length = 512
        self.embedding_dimension = 1024

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on GPU and in FP32
        Q = Q.to(torch.float32).contiguous()
        K = K.to(torch.float32).contiguous()
        V = V.to(torch.float32).contiguous()

        # Compute QK^T
        attn = self.matmul(Q, K, self.sequence_length, self.embedding_dimension)
        # Apply softmax
        attn = self.softmax(attn, self.sequence_length)
        # Multiply with V
        out = self.matmul(attn, V, self.sequence_length, self.embedding_dimension)
        return out

    @triton.jit
    def matmul_kernel(
        Q_ptr, K_ptr, out_ptr,
        batch_size, num_heads, seq_len, dim,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        batch = pid // num_heads
        head = pid % num_heads

        offset_q = batch * num_heads * seq_len * dim + head * seq_len * dim
        offset_k = offset_q
        offset_out = batch * num_heads * seq_len * seq_len + head * seq_len * seq_len

        block_start = tl.program_id(1) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len

        q = tl.load(Q_ptr + offset_q + offsets, mask=mask, other=0.0)
        k = tl.load(K_ptr + offset_k + offsets, mask=mask, other=0.0)
        out = tl.dot(q, k)
        tl.store(out_ptr + offset_out + offsets, out, mask=mask)

    def matmul(self, Q, K, seq_len, dim):
        out = torch.empty(batch_size, num_heads, seq_len, seq_len, device='cuda', dtype=torch.float32)
        grid = (num_heads * batch_size, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
        self.matmul_kernel[grid](Q, K, out, batch_size, self.num_heads, seq_len, dim, BLOCK_SIZE=128)
        return out

    @triton.jit
    def softmax_kernel(
        input_ptr, output_ptr,
        batch_size, num_heads, seq_len,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        batch = pid // num_heads
        head = pid % num_heads

        offset_input = batch * num_heads * seq_len * seq_len + head * seq_len * seq_len
        offset_output = offset_input

        block_start = tl.program_id(1) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len

        input = tl.load(input_ptr + offset_input + offsets, mask=mask, other=-float('inf'))
        max_val = tl.max(input, axis=0)
        input -= max_val
        exp_input = tl.exp(input)
        sum_exp = tl.sum(exp_input, axis=0)
        softmax = exp_input / sum_exp
        tl.store(output_ptr + offset_output + offsets, softmax, mask=mask)

    def softmax(self, input, seq_len):
        out = torch.empty_like(input)
        grid = (num_heads * batch_size, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
        self.softmax_kernel[grid](input, out, batch_size, self.num_heads, seq_len, BLOCK_SIZE=128)
        return out

    @triton.jit
    def matmul_v_kernel(
        attn_ptr, V_ptr, out_ptr,
        batch_size, num_heads, seq_len, dim,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        batch = pid // num_heads
        head = pid % num_heads

        offset_attn = batch * num_heads * seq_len * seq_len + head * seq_len * seq_len
        offset_v = batch * num_heads * seq_len * dim + head * seq_len * dim
        offset_out = batch * num_heads * seq_len * dim + head * seq_len * dim

        block_start = tl.program_id(1) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len

        attn = tl.load(attn_ptr + offset_attn + offsets, mask=mask, other=0.0)
        v = tl.load(V_ptr + offset_v + offsets * dim, mask=mask, other=0.0)
        out = tl.dot(attn, v)
        tl.store(out_ptr + offset_out + offsets, out, mask=mask)

    def matmul(self, attn, V, seq_len, dim):
        out = torch.empty(batch_size, num_heads, seq_len, dim, device='cuda', dtype=torch.float32)
        grid = (num_heads * batch_size, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
        self.matmul_v_kernel[grid](attn, V, out, batch_size, self.num_heads, seq_len, dim, BLOCK_SIZE=128)
        return out