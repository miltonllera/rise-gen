import numpy.random as npr
import torch
from torch.utils.data import Dataset
from synthetic.sample import StarRobot


class PlaceholderDataset(Dataset):
    def __init__(
        self,
        num: int,
    ):
        self.num = num

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return idx


class StarRobotDataset(Dataset):
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        min_num_nodes: int,
        max_num_nodes: int,
        grid_size: int,
        seed: int | None = None
    ) -> None:
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes
        self.grid_size = grid_size
        self.rng = npr.default_rng(seed)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, _) -> torch.Tensor:
        return StarRobot(
            self.min_num_nodes,
            self.max_num_nodes,
            batch_size=self.batch_size,
            device='cpu',  # type: ignore
            seed=self.rng.integers(0, 2**32-1),  # type: ignore
            resolution=self.grid_size,
        ).get().squeeze()
