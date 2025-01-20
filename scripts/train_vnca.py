import argparse
import torch as t
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from model.vae.vnca import VNCA
from synthetic.dataset import StarRobotDataset


GRID_SIZE=64

robot_dataset = StarRobotDataset(
    dataset_size=10000000,
    batch_size=8,
    min_num_nodes=3,
    max_num_nodes=8,
    grid_size=GRID_SIZE,
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

vae = VNCA(
    z_dim=512,
    nca_hid=128,
    max_num_nodes=8,
    grid_size=GRID_SIZE,
    vrn_dim=32,
    vrn_depth=5,
    conv_layers=3,
)

vae.hparams.lr = 3e-5
vae.beta = 1
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
    devices=[0, 1, 2, 3],
    check_val_every_n_epoch=1,
    max_epochs=-1,
)
t.set_float32_matmul_precision("high")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training Variational NCAs")

    parser.add_argument("--grid_size", type=int, default=64,
        help="Maximum morphology size along each dimension")
    parser.add_argument("--update-net-depth", type=int, default=4,
        help="Depth of the feed-forward network used to update the NCA cell states.")
    parser.add_argument("--nca_hid", type=int, default=128,
        help="Size of the hidden state used in the NCA decoder update function")

    trainer.fit(vae, train_loader, valid_loader)
