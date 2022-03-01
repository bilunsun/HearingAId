import argparse
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from tqdm.auto import tqdm

from lib import AudioDataModule, EnvNet


class Scaler(nn.Module):
    """Basic log scaler class"""

    def __init__(self, size: int):
        super().__init__()

        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("std", torch.ones(size) / 5)
        self.device = torch.device("cpu")

    def fit(self, x):
        mask = x != 0
        self.mean = torch.mean(x[mask].view(-1))
        self.std = torch.std(x[mask].view(-1))

    def transform(self, x):
        return (x - self.mean) / self.std

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.device = device

    def __repr__(self):
        return f"mean: {self.mean}\tstd: {self.std}"


class PretrainSimSiam(pl.LightningModule):
    """
    Defines the hooks for training using pytorch lightning
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--n_classes", type=int, default=10)
        parser.add_argument("--transfer_ckpt", type=str, default=None)
        parser.add_argument("--width", type=int, default=16_000)
        parser.add_argument("--hidden_dim", type=int, default=1024)
        parser.add_argument("--out_dim", type=int, default=1024)
        parser.add_argument("--lr", type=float, default=5e-4)

        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.backbone = EnvNet(n_classes=self.hparams.n_classes)
        self.backbone.classifier = nn.Identity()

        with torch.no_grad():
            x = torch.randn(1, 1, 1, self.hparams.width)
            self.latent_dim = self.backbone(x).size(-1)

        self.projection_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.hparams.hidden_dim),
            nn.BatchNorm1d(self.hparams.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.BatchNorm1d(self.hparams.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hparams.hidden_dim, self.hparams.out_dim),
            nn.BatchNorm1d(self.hparams.hidden_dim)
        )

        # Encoder
        self.f = nn.Sequential(
            self.backbone,
            self.projection_mlp
        )

        # Predictor
        self.h = nn.Sequential(
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.BatchNorm1d(self.hparams.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hparams.hidden_dim, self.hparams.out_dim)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        self.criterion = nn.KLDivLoss(reduction="batchmean")

        self.scaler = Scaler(size=())

    def criterion(self, p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, _):
        x1, x2 = batch
        z1, z2 = self.f(x1), self.f(x2)
        p1, p2 = self.h(z1), self.h(z2)

        loss = self.criterion(p1, z2) / 2 + self.criterion(p2, z1) / 2

        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, _):
        x1, x2 = batch
        z1, z2 = self.f(x1), self.f(x2)
        p1, p2 = self.h(z1), self.h(z2)

        loss = self.criterion(p1, z2) / 2 + self.criterion(p2, z1) / 2

        self.log("val_loss", loss, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return self.optimizer

    def on_fit_start(self):
        all_x = []
        for i, (x1, x2) in enumerate(tqdm(self.trainer.datamodule.train_dataloader())):
            if i == 5:
                break
            all_x.append(x1)
            all_x.append(x2)
        all_x = torch.cat(all_x)

        self.scaler.fit(all_x)
        self.scaler.to("cuda")
        print("mean.shape", self.scaler.mean.shape, "mean", self.scaler.mean)
        print("std.shape", self.scaler.std.shape, "std", self.scaler.std)

        self.class_names = self.trainer.datamodule.class_names

        self.logger.log_hyperparams(
            {
                "train_set_size": len(self.trainer.datamodule.train_dataset),
                "val_set_size": len(self.trainer.datamodule.val_dataset),
                "train_folds": self.trainer.datamodule.train_folds,
                "val_folds": self.trainer.datamodule.val_folds,
                "class_names": self.class_names,
            }
        )


class WandbModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        pl_module.logger.experiment.save(self.best_model_path, base_path=self.dirpath)


def main(args):
    configs = vars(args)

    # Set the seed; pl.seed_everything sets a random seed if args.seed is None
    seed = pl.seed_everything(configs["seed"])
    configs["seed"] = seed

    # Pretrain
    configs["pretraining"] = "simsiam"

    audio_datamodule = AudioDataModule(**configs)
    configs["width"] = audio_datamodule.width
    configs["height"] = audio_datamodule.height
    configs["n_classes"] = audio_datamodule.n_classes
    model = PretrainSimSiam(**configs)

    print("Model hparams")
    print(model.hparams)

    logger = WandbLogger(project=configs["project"])
    logger.experiment.watch(model, log="all")

    checkpoint_callback = ModelCheckpoint(
        dirpath=configs["checkpoints_dir"],
        filename=logger.experiment.name,
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,
        verbose=True,
    )

    callbacks = [checkpoint_callback]

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger,
        callbacks=callbacks,
        max_epochs=configs["max_epochs"],
        check_val_every_n_epoch=configs["check_val_every_n_epoch"],
    )

    trainer.fit(model, datamodule=audio_datamodule, ckpt_path="checkpoints/scarlet-leaf-119-v1.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="urbansound8k")
    parser.add_argument("--target_sample_rate", type=int, default=16_000)
    parser.add_argument("--n_samples", type=int, default=24_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--max_epochs", type=int, default=5_000)
    parser.add_argument("--project", type=str, default="TEST-hearingAId")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)

    parser = PretrainSimSiam.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
