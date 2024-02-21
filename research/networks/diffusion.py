# U-Net implementation from: Diffusion Policy Codebase (Chi et al; arXiv:2303.04137)

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Callable, Optional, Union

import gym
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer("emb", emb)

    def forward(self, x):
        # X is shape (Batch dims,) Emb is shape (D)
        pos_emb = x.unsqueeze(-1) * self.emb
        return torch.cat((pos_emb.sin(), pos_emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1)))

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=3,
        n_groups=8,
    ):
        """
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
            The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """
        super().__init__()
        assert isinstance(action_space, gym.spaces.Box)
        assert len(action_space.shape) == 1
        input_dim = action_space.shape[0]
        assert isinstance(observation_space, gym.spaces.Box)
        assert len(observation_space.shape) == 1
        global_cond_dim = observation_space.shape[0]

        all_dims = [input_dim, *list(down_dims)]
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                ),
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of diffusion parameters: {:e}".format(sum(p.numel() for p in self.parameters())))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, int],
        cond: Optional[torch.Tensor] = None,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B, cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif isinstance(timesteps, torch.Tensor) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        assert isinstance(timesteps, torch.Tensor)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature: torch.Tensor = self.diffusion_step_encoder(timesteps)

        if cond is not None:
            global_feature = torch.cat([global_feature, cond], dim=-1)

        x = sample
        h = []
        for _idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for _idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x


class MLPResNetBlock(nn.Module):
    def __init__(self, dim: int, use_layer_norm: bool = False, dropout: float = 0.0, act: Callable = nn.Mish):
        super().__init__()
        net = []
        if dropout > 0:
            net.append(nn.Dropout(dropout))
        if use_layer_norm:
            net.append(nn.LayerNorm(dim))
        net.extend([nn.Linear(dim, 4 * dim), act(), nn.Linear(4 * dim, dim)])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return x + self.net(x)


class MLPResNet(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        num_blocks: int = 3,
        time_dim: int = 64,
        hidden_dim=256,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        act=nn.Mish,
    ):
        super().__init__()
        self.time_net = nn.Sequential(
            SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim), act(), nn.Linear(time_dim, time_dim)
        )
        act_dim = action_space.shape[0]
        cond_dim = observation_space.shape[0]
        self.proj = nn.Linear(act_dim + cond_dim + time_dim, hidden_dim)
        self.net = nn.Sequential(
            *(
                MLPResNetBlock(hidden_dim, dropout=dropout, use_layer_norm=use_layer_norm, act=act)
                for _ in range(num_blocks)
            )
        )
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, act_dim))

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor):
        x = torch.cat((sample, self.time_net(timestep), cond), axis=-1)
        x = self.proj(x)
        x = self.net(x)
        x = self.out(x)
        return x.view(sample.shape)
