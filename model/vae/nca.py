import torch
import torch.nn as nn
from typing import Callable

from .ca import CellularAutomata
from .nn import Residual


class VoxelPerception(nn.Conv3d):
    def __init__(self, in_channels):
        super().__init__(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )


class VoxelUpdate(nn.Module):
    def __init__(
        self,
        n_channels: int,
        hidden_size: int,
        n_layers: int = 2,
        act_fn: Callable | nn.Module = nn.ELU,
        p_update: float = 1.0,
        alive_threshold: float | None = None,
    ) -> None:
        super().__init__()

        assert 0 < p_update <= 1.0, "Update probability must be greater than 0 and less than 1"

        if alive_threshold is not None:
            alive_fun = VoxelAliveFunction(alive_threshold)
        else:
            alive_fun = nn.Identity()

        self.update_net = _create_conv_update_network(n_channels, hidden_size, n_layers, act_fn)
        self.alive_fun = alive_fun
        self.p_update = nn.Dropout(p_update)


    def forward(self, state, perception, condition=None):
        if condition is not None:
            perception = torch.cat([perception, condition], dim=1)

        pre_alive_mask = self.alive_fun(state)

        update = self.update_net(perception)
        update = self.p_update(update) * update
        state = state + update

        post_alive_mask = self.alive_fun(state)

        return state * (pre_alive_mask & post_alive_mask)


class VoxelAliveFunction(nn.Module):
    def __init__(self, threshold: float, alive_index: int = 0):
        assert 0 < threshold, "Alive threshold must be greater than 0"
        self.threshold = threshold
        self.alive_index = torch.tensor([alive_index])
        self.max_pool = nn.MaxPool3d(kernel_size=3, padding=1)

    def forward(self, state):
        return self.max_pool(state[:, self.alive_index]) > self.threshold


class VoxelNCADecoder(nn.Module):
    def __init__(
        self,
        latent_size: int,
        output_size: int,
        n_doubling_steps: int,
        nca: CellularAutomata,
        init_resolution: int = 2,
        init_fn: Callable | None = None,
        use_position_embeddings: bool = False,
        condition_nca: bool = False,
    ) -> None:
        super().__init__()

        position_embeddings = torch.zeros((init_resolution * 3, latent_size))
        if use_position_embeddings:
            position_embeddings = torch.nn.init.normal_(position_embeddings, std=0.1)
            position_embeddings = nn.Parameter(position_embeddings)

        if init_fn is None:
            def init_fn(z, pe):
                z = torch.tile(z.unsqueeze(1), (1, init_resolution * 3, 1))
                init = (z + pe).movedim(1, -1)
                return init.unflatten(1, ([init_resolution * 3]))

        self.n_doubling_steps = n_doubling_steps
        self.init_resolution = init_resolution
        self.position_embeddings = position_embeddings
        self.init_fn = init_fn
        self.nca = nca
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.output_fun = nn.Linear(latent_size, output_size, bias=True)
        self.condition_nca = condition_nca

    def forward(self, z: torch.Tensor):
        state = self.init_fn(z, self.position_embeddings)
        condition = z if self.condition_nca else None

        for m in range(self.n_doubling_steps):
            state, _ = self.nca(state, condition=condition, n_steps=2*m+2)
            state = self.upsample(state)

        return self.output_fun(state.movedim(1, -1)).movedim(-1, 1)


def _create_conv_update_network(
    input_dim: int,
    hidden_dim: int,
    n_layers: int,
    act_fn: Callable | nn.Module = nn.ELU,
):

    res_block = _create_conv_residual_block(
        input_dim, hidden_dim, hidden_dim, act_fn
    )
    layers: list[nn.Module] = [res_block]

    for _ in range(n_layers-1):
        res_block = _create_conv_residual_block(
            hidden_dim, hidden_dim, hidden_dim, act_fn
        )
        layers.append(res_block)

    layers.append(nn.Conv3d(hidden_dim, input_dim, kernel_size=1))

    return nn.Sequential(*layers)


def _create_conv_residual_block(input_dim, hidden_dim, output_dim, act_fn):
    layer_block = [
        nn.Conv3d(input_dim, hidden_dim, kernel_size=1, bias=False),
        act_fn(),
        nn.Conv3d(hidden_dim, output_dim, kernel_size=1, bias=False),
    ]
    return Residual(*layer_block)
