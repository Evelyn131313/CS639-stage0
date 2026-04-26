import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transposed_conv3d(x)

    def transposed_conv3d(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on CUDA
        assert x.is_cuda, "Input tensor must be on CUDA."
        # Ensure kernel and bias are on CUDA
        assert self.kernel.is_cuda, "Kernel weights must be on CUDA."
        if self.bias:
            assert self.bias_tensor.is_cuda, "Bias tensor must be on CUDA."

        # Prepare output tensor
        batch_size, in_channels, depth_in, height_in, width_in = x.shape
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        output_padding_d, output_padding_h, output_padding_w = self.output_padding

        # Compute output dimensions
        depth_out = (depth_in - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d
        height_out = (height_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h
        width_out = (width_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w

        out = torch.empty((batch_size, self.out_channels, depth_out, height_out, width_out), dtype=x.dtype, device=x.device)

        # Launch Triton kernel
        self._launch_kernel(x, out, self.kernel, self.bias_tensor if self.bias else None, depth_in, height_in, width_in, depth_out, height_out, width_out, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, output_padding_d, output_padding_h, output_padding_w)
        return out

    def _launch_kernel(self, x: torch.Tensor, out: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor, depth_in, height_in, width_in, depth_out, height_out, width_out, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, output_padding_d, output_padding_h, output_padding_w):
        # Ensure tensors are contiguous
        x = x.contiguous()
        out = out.contiguous()
        kernel = kernel.contiguous()
        if bias is not None:
            bias = bias.contiguous()

        # Determine block size
        BLOCK_SIZE = 128
        num_elements = depth_out * height_out * width_out * self.out_channels
        num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Launch kernel
        transposed_conv3d_kernel[triton.cudevice(), (num_blocks, 1, 1)](
            x, kernel, out, bias,
            depth_in, height_in, width_in, depth_out, height_out, width_out,
            kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w, output_padding_d, output_padding_h, output_padding_w,
            BLOCK_SIZE
        )

    @triton.jit
    def transposed_conv3d_kernel(
        x_ptr, kernel_ptr, out_ptr, bias_ptr,
        depth_in, height_in, width_in, depth_out, height_out, width_out,
        kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w, output_padding_d, output_padding_h, output_padding_w,
        BLOCK_SIZE: tl.constexpr
    ):
        # Each program handles a contiguous block of data
        pid = tl.program_id(0)
        offset = pid * BLOCK_SIZE
        offsets = tl.arange(0, BLOCK_SIZE)

        # Compute the output index
        out_idx = offset + offsets
        out_b, out_c, out_d, out_h, out_w = tl.packbits(out_idx, 5)
        out_b = out_b // (depth_out * height_out * width_out * self.out_channels)
        out_c = (out_b * depth_out * height_out * width_out + out_idx) // (depth_out * height_out * width_out)
        out_d = (out_c * height_out * width_out + out_idx) // (height_out * width_out)
        out_h = (out_d * width_out + out_idx) // width_out
        out_w = out_idx % width_out

        # Compute the input indices
        in_d = (out_d - output_padding_d) // stride_d + padding_d - (kernel_d - 1) // 2
        in_h = (out_h - output_padding_h) // stride_h + padding_h - (kernel_h - 1) // 2
        in_w = (out_w - output_padding_w) // stride_w + padding_w - (kernel_w - 1) // 2

        # Accumulate the result
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    in_idx = (out_b * in_channels * depth_in * height_in * width_in) + (out_c * depth_in * height_in * width_in) + (in_d * height_in * width_in) + (in_h * width_in) + in_w
                    kernel_idx = (out_c * in_channels * kernel_d * kernel_h * kernel_w) + (out_b * kernel_d * kernel_h * kernel_w) + (kd * kernel_h * kernel_w) + (kh * kernel_w) + kw
                    x_val = tl.load(x_ptr + in_idx, mask=out_idx < depth_out * height_out * width_out * self.out_channels, other=0.0)
                    kernel_val = tl.load(kernel_ptr + kernel_idx, mask=out_idx < depth_out * height_out * width_out * self.out_channels, other=0.0)
                    acc += x_val * kernel_val

        # Apply bias
        if bias_ptr is not None:
            bias_idx = out_c
            bias_val = tl.load(bias_ptr + bias_idx, mask=out_idx < depth_out * height_out * width_out * self.out_channels, other=0.0)
            acc += bias_val

        # Store the result
        tl.store(out_ptr + out_idx, acc, mask=out_idx < depth_out * height_out * width_out * self.out_channels)