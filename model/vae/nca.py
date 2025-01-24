from typing import Callable
import torch
import torch.nn as nn


class NCA(nn.Module):
    def __init__(
        self,
        update_net: nn.Module,
        min_steps: int,
        max_steps: int | None = None,
        p_update: float = 1.0,
        alive_mask: Callable | None = None,
        norm_update: Callable | None = None,
    ) -> None:
        super().__init__()

        if max_steps is None:
            max_steps = min_steps
            min_steps = None  # type: ignore

        if min_steps is not None and max_steps < min_steps:
            raise RuntimeError(
                f"max_steps must be greater than min_steps, got {max_steps} and {min_steps}"
            )

        assert 0.0 <= p_update <= 1.0, "Update probability must be in the range [0, 1]"

        if norm_update is None:
            norm_update = nn.Identity()

        self.update_net = update_net
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.p_update = p_update
        self.alive_mask = alive_mask
        self.norm_update = norm_update

    def forward(self, inputs, n_steps=None, return_trajectory=False):
        B = inputs.shape[0]

        if n_steps is None:
            n_steps = self.max_steps

        state = inputs
        batch_steps = _broadcast(self.sample_steps(B, n_steps), inputs)

        i, not_done = 0, 0 < batch_steps
        trajectory = [state] if return_trajectory else None

        while torch.any(not_done) and i < n_steps:
            pre_alive_mask = self.is_alive(state)

            update = self.update_net(state)
            update_mask = self.update_mask(state) & not_done
            state = state + self.norm_update(update_mask * update)

            post_alive_mask = self.is_alive(state)
            state = state * (pre_alive_mask & post_alive_mask)

            if trajectory is not None:
                trajectory.append(state)

            # update iteration
            i += 1
            not_done = i < batch_steps

        return state, trajectory

    def update_mask(self, state):
        if self.p_update < 1.0:
            return _broadcast(torch.rand((len(state),)) < self.p_update, state)
        return True

    def is_alive(self, state):
        if self.alive_mask is not None:
            return self.alive_mask(state)
        return True

    def sample_steps(self, batch_size, n_steps=None):
        if n_steps is None:
            n_steps = self.max_steps

        if self.min_steps is None or self.min_steps > n_steps:
            return n_steps
        return torch.randint(self.min_steps, n_steps, (batch_size,))


class NCADecoder(nn.Module):
    def __init__(
        self,
        latent_size: int,
        output_size: int,
        nca: NCA,
        init_resolution: int,
        n_dims: int,
        n_doubling_steps: int,
    ) -> None:
        super().__init__()

        self.init_resolution = init_resolution
        self.n_dims = n_dims
        self.n_doubling_steps = n_doubling_steps
        self.nca = nca
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.output_fun = nn.Linear(latent_size, output_size, bias=False)

    def forward(self, z):
        # broadcast z to all initial grid cells
        state = torch.tile(
            z.view(*z.shape, *([1] * self.n_dims)),
            ([self.init_resolution] * self.n_dims)
        )

        for m in range(self.n_doubling_steps):
            state, _ = self.nca(state, n_steps=2*m+2)
            state = self.upsample(state)

        return self.output_fun(state.movedim(1, -1)).movedim(-1, 1)


def _broadcast(mask, tensor):
    mask_shape = [len(tensor)] + [1] * (tensor.ndim - 1)
    return mask.view(*mask_shape).to(device=tensor.device)
