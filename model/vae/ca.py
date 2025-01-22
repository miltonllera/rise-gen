from typing import Callable
import torch
import torch.nn as nn


class CellularAutomata(nn.Module):
    def __init__(
        self,
        perception_fn: Callable,
        update_fn: Callable,
        max_steps: int,
        min_steps: int | None = None,
        input_net: Callable = nn.Identity(),
    ) -> None:
        super().__init__()

        assert min_steps is None or min_steps <= max_steps, \
            "min_steps must be less or equal than max_steps"

        self.min_steps = min_steps
        self.max_steps = max_steps
        self.perception_fn = perception_fn
        self.update_fn = update_fn
        self.input_net = input_net

    def forward(self, state, inputs=None, n_steps=None, return_trajectory=False):
        B = len(state)

        condition = self.input_net(inputs) if inputs is not None else None

        if n_steps is None:
            n_steps = self.max_steps

        batch_steps = _broadcast(self.sample_steps(B, n_steps), state)

        i, not_done = 0, 0 < batch_steps
        trajectory = [state] if return_trajectory else None

        while torch.any(not_done) and (i := i + 1) < n_steps:
            percpetion = self.perception_fn(state)
            update = self.update_fn(state, percpetion, condition)
            state = state + update * not_done

            if return_trajectory:
                trajectory.append(state)  # type: ignore

            # update iteration
            not_done = i <= batch_steps

        return state, trajectory

    def sample_steps(self, batch_size, n_steps=None):
        if n_steps is None:
            n_steps = self.max_steps

        if self.min_steps is None or self.min_steps >= n_steps:
            return n_steps

        return torch.randint(self.min_steps, n_steps, (batch_size,))


def _broadcast(mask, tensor):
    mask_shape = [len(tensor)] + [1] * (tensor.ndim - 1)
    return mask.view(*mask_shape).to(device=tensor.device)
