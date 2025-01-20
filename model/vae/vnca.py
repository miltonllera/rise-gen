import math
from typing import Callable, Literal
import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from synthetic.sample import StarRobot
from .encoder import RobotConvEncoder
from .nca import NCA, NCADecoder
from .nn import Residual, DiagonalGaussian


class VNCA(pl.LightningModule):
    def __init__(
        self,
        z_dim: int,
        grid_size: int,
        max_num_nodes: int,
        vrn_dim: int,
        vrn_depth: int,
        conv_layers: int,
        state_dim: int | None = None,
        nca_hid: int | None = None,
        nca_layers: int = 4,
        init_resolution: int=2,
        position_dependent_cell_init: bool = False,
        condition_nca: bool = False,
        beta: float = 0.1,
        lr: float = 3e-5,
        step_size: int = 4,
        gamma: float = 0.91,
    ):
        super().__init__()

        n_doubling_steps = int(math.log2(grid_size / init_resolution))
        assert math.pow(2, n_doubling_steps) * init_resolution == grid_size

        if state_dim is None:
            state_dim = z_dim

        if nca_hid is None:
            nca_hid = 4 * state_dim

        self.save_hyperparameters()  # saves parameters automatically using reflection
        self.beta = beta

        # NOTE: For now use the same encoder as the original model
        self.encoder = RobotConvEncoder(
            f_dim=max_num_nodes + 2,
            e_dim=z_dim,
            grid_size=grid_size,
            vrn_dim=vrn_dim,
            vrn_depth=vrn_depth,
            n_conv_encoder_layers=conv_layers,
        )
        self.latent = DiagonalGaussian(z_dim, z_dim)
        self.decoder = NCADecoder(
            latent_size=state_dim,
            output_size=max_num_nodes + 2,
            nca=NCA(
                update_net=create_conv_update_network(
                    state_dim,
                    nca_hid,
                    n_layers=nca_layers,
                    n_dim=3,
                    act_fn=nn.ELU,
                    residual=True
                ),
                min_steps=0,
                max_steps=20,
            ),
            init_resolution=init_resolution,
            n_dims=3,
            n_doubling_steps=n_doubling_steps,
            use_position_embeddings=position_dependent_cell_init,
            condition_nca=condition_nca,
        )

    def get_samples(self, n, seed: int | None = None):
        samples = StarRobot(
            self.hparams.min_num_nodes,  # type: ignore
            self.hparams.max_num_nodes,  # type: ignore
            n,
            self.device,  # type: ignore
            seed,  # type: ignore
            self.hparams.grid_size,  # type: ignore
        ).get()
        return samples

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.latent(mu, logvar)
        x_ = self.decoder(z)
        return x_, z, (mu, logvar)

    def training_step(self, x, batch_idx):
        x_, _, (mu, logvar) = self.forward(x)

        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        recon_loss = self.decoder_loss(x, x_)
        loss = recon_loss + self.beta * kld_loss

        self.log(
            "reconstruction_loss",
            recon_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "kld_loss",
            kld_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, x, batch_idx):
        x_, _, (mu, logvar) = self.forward(x)

        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kld_loss = kld.mean()
        recon_loss = self.decoder_loss(x, x_)
        self.log(
            "val_reconstruction_loss",
            recon_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_kld_loss",
            kld_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        metrics = {"val_loss": recon_loss}
        self.log_dict(metrics, sync_dist=True)
        return metrics

    def on_train_epoch_end(self):
        self.beta = min(self.beta * 1.12, 1)

    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
        )
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

    def decoder_loss(self, x, x_):
        labels = torch.argmax(x, dim=1)
        ce_loss = nn.CrossEntropyLoss()(x_, labels)
        cew_loss = nn.CrossEntropyLoss(weight=self.get_class_weight(x))(x_, labels)
        return ce_loss + cew_loss

    def generate_by_latent(self, latent):
        self.decoder.eval()
        with torch.no_grad():
            if type(latent) is not torch.Tensor:
                latent = torch.tensor(np.array(latent), dtype=torch.float32).to(self.device)
            new_samples = self.decode(latent)
        return new_samples

    def generate(self, batch_size, seed=None, mean=None, var=None):
        if seed is None:
            latent = torch.randn([batch_size, self.hparams.z_dim], device=self.device)
        else:
            generator = torch.Generator(self.device).manual_seed(seed)
            latent = torch.randn(
                [batch_size, self.hparams.z_dim],
                generator=generator,
                device=self.device,
            )
        if mean is not None and var is not None:
            latent = latent * var + mean
        return self.generate_by_latent(latent)

    @staticmethod
    def get_class_weight(target_segmentation):
        seg_num = torch.sum(target_segmentation, dim=(0, 2, 3, 4))
        weight = torch.where(seg_num > 0, 1 / seg_num, 0)
        return weight / weight.sum()


def create_conv_update_network(
    input_dim: int,
    hidden_dim: int,
    n_layers: int,
    n_dim: Literal[2] | Literal[3] = 3,
    act_fn: Callable | nn.Module = nn.ELU,
    residual: bool = True,
):

    if n_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    layers: list[nn.Module] = [conv(input_dim, hidden_dim, kernel_size=3, padding=1)]
    for _ in range(n_layers):
        layer_block = [
            conv(hidden_dim, hidden_dim, kernel_size=1),
            act_fn(),
            conv(hidden_dim, hidden_dim, kernel_size=1),
        ]

        if residual:
            layers.append(Residual(*layer_block))
        else:
            layers.extend(layer_block)

    layers.append(conv(hidden_dim, input_dim, kernel_size=1))

    return nn.Sequential(*layers)
