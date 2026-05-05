import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_forward_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    mean_ptr,
    rsqrt_ptr,
    n_elements_per_group,
    n_groups,
    BLOCK_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    block_id = tl.program_id(1)
    block_start = group_id * n_elements_per_group + block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_per_group

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + group_id, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + group_id, mask=mask, other=0.0)

    mean = tl.load(mean_ptr + group_id, mask=mask, other=0.0)
    rsqrt = tl.load(rsqrt_ptr + group_id, mask=mask, other=1.0)

    x_centered = x - mean
    x_normalized = x_centered * rsqrt
    out = gamma * x_normalized + beta

    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        self.register_buffer('mean', torch.zeros(num_groups))
        self.register_buffer('rsqrt', torch.ones(num_groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_elements_per_group = self.num_features // self.num_groups
        BLOCK_SIZE = 256  # Tunable parameter for block size

        grid = lambda meta: (
            self.num_groups,
            (n_elements_per_group + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        )

        group_norm_forward_kernel[grid](
            x,
            self.gn.weight,
            self.gn.bias,
            x,
            self.mean,
            self.rsqrt,
            n_elements_per_group,
            self.num_groups,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return x


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features, num_groups]  # num_features