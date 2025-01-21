import argparse
import torch as t
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from model.vae.vnca import VNCA
from synthetic.dataset import StarRobotDataset


def create_dataset(min_nodes, max_nodes, grid_size, batch_size):
    robot_dataset = StarRobotDataset(
        dataset_size=10000000,
        batch_size=batch_size,
        min_num_nodes=min_nodes,
        max_num_nodes=max_nodes,
        grid_size=grid_size,
    )

    train_size = int(0.9 * len(robot_dataset))
    valid_size = len(robot_dataset) - train_size

    train_dataset, valid_dataset = random_split(robot_dataset, [train_size, valid_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=7,
        shuffle=False,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=8,
        num_workers=7,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, valid_loader


def create_model(
    max_nodes,
    grid_size,
    z_dim,
    state_dim,
    update_net_layers,
    position_dependent_cell_init,
    condition_nca,
):
    vnca = VNCA(
        z_dim=z_dim,
        state_dim=state_dim,
        max_num_nodes=max_nodes,
        grid_size=grid_size,
        vrn_dim=32,
        vrn_depth=5,
        conv_layers=3,
        n_update_net_layers=update_net_layers,
        position_dependent_cell_init=position_dependent_cell_init,
        condition_nca=condition_nca,
    )

    vnca.hparams.lr = 3e-5
    vnca.beta = 1

    return vnca


def main(
    grid_size: int,
    latent_size: int,
    update_net_layers: int,
    position_dependent_cell_init: bool,
    condition_nca: bool,
    state_dim: int | None = None,
    batch_size: int = 8,
    min_nodes: int = 3,
    max_nodes: int = 8,
    devices=[0, 1, 2, 3]
):

    train_loader, valid_loader = create_dataset(min_nodes, max_nodes, grid_size, batch_size)
    vnca = create_model(
        max_nodes,
        grid_size,
        latent_size,
        state_dim,
        update_net_layers,
        position_dependent_cell_init,
        condition_nca,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="data/ckpt/vnca/",
        filename="v2-{epoch:02d}-{val_loss:.3f}",
        save_top_k=10,
        verbose=True,
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,
    )
    logger = TensorBoardLogger(save_dir="logger")
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        limit_train_batches=4096,
        limit_val_batches=512,
        strategy="ddp",
        devices=devices,
        check_val_every_n_epoch=1,
        max_epochs=-1,
    )
    t.set_float32_matmul_precision("high")

    trainer.fit(vnca, train_loader, valid_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training Variational NCAs")

    parser.add_argument("--grid_size", type=int, default=64,
        help="Maximum morphology size along each dimension")
    parser.add_argument("--latent_size", type=int, default=512,
        help="Size of the latent space used in the VNCA.")
    parser.add_argument("--state_dim", type=int, default=32,
        help="State size of the NCA cells")
    parser.add_argument("--update_net_depth", type=int, default=4,
        help="Depth of the feed-forward network used to update the NCA cell states.")
    parser.add_argument("--positional_init", action='store_true', type=bool,
        help="Whether to use positional information to condition the initial state init.")
    parser.add_argument("--condition_nca", action='store_true', type=bool,
        help="Conditon the NCA' update net on the latent vector at each time step.")
    parser.add_argument("--nca_hid", type=int, default=128,
        help="Size of the hidden state used in the NCA decoder update function")


    args = parser.parse_args()

    main(**vars(args))
