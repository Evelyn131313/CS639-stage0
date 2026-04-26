import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.triton_avg_pool(x)

    def triton_avg_pool(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input must be on CUDA."
        x = x.contiguous()
        batch_size, channels, height, width = x.shape
        output_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = torch.empty((batch_size, channels, output_height, output_width), dtype=x.dtype, device=x.device)

        @triton.jit
        def avg_pool_kernel(
            input_ptr, 
            output_ptr, 
            kernel_size: tl.constexpr, 
            padding: tl.constexpr, 
            stride: tl.constexpr, 
            batch_size: tl.constexpr, 
            channels: tl.constexpr, 
            height: tl.constexpr, 
            width: tl.constexpr, 
            output_height: tl.constexpr, 
            output_width: tl.constexpr, 
            BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(0)
            i = pid // output_width
            j = pid % output_width
            batch_idx = pid // (output_height * output_width)
            channel_idx = (pid % (output_height * output_width)) // output_width

            start_h = i * stride - padding
            start_w = j * stride - padding

            # Compute the block index within the window
            block_idx = tl.program_id(1)
            block_h = block_idx // output_width
            block_w = block_idx % output_width

            # Compute the thread index within the block
            thread_idx = tl.program_id(2)
            dh = thread_idx // kernel_size
            dw = thread_idx % kernel_size

            h = start_h + dh
            w = start_w + dw

            # Compute input and output indices
            input_idx = batch_idx * channels * height * width + channel_idx * height * width + h * width + w
            output_idx = batch_idx * channels * output_height * output_width + channel_idx * output_height * output_width + i * output_width + j

            input_val = tl.load(input_ptr + input_idx, mask=(h >= 0) & (h < height) & (w >= 0) & (w < width), other=0.0)
            tl.atomic_add(output_ptr + output_idx, input_val)

        # Launch the kernel
        grid = (batch_size * channels * output_height * output_width, 1, 1)
        avg_pool_kernel[grid](x, output, self.kernel_size, self.padding, self.stride, batch_size, channels, height, width, output_height, output_width, self.kernel_size * self.kernel_size)

        # Final normalization
        output /= (self.kernel_size * self.kernel_size)
        return output