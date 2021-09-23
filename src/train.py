import argparse
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lib import AudioDataModule, VanillaCNN


class Model(pl.LightningModule):
    """
    Defines the hooks for training using pytorch lightning
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--width", type=int, default=64)
        parser.add_argument("--height", type=int, default=64)

        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--scheduler", type=str, choices=[None, "cosine", "plateau"], default=None)
        parser.add_argument("--min_lr", type=float, default=1e-7)
        parser.add_argument("--plateau_factor", type=float, default=0.1)
        parser.add_argument("--plateau_patience", type=int, default=10)
        parser.add_argument("--plateau_cooldown", type=int, default=0)

        return parser

    def __init__(self, lr, scheduler, min_lr, plateau_factor, plateau_patience, plateau_cooldown, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.model = VanillaCNN(self.hparams.width, self.hparams.height)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        if self.hparams.scheduler is None:
            self.scheduler = None
        elif self.hparams.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr
            )
        else:  # Choices are [None, "cosine", "plateau"] from argparse
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.hparams.plateau_factor,
                patience=self.hparams.plateau_patience,
                min_lr=self.hparams.min_lr,
                cooldown=self.hparams.plateau_cooldown,
            )

    def forward(self, x):
        return self.model(x)

    def _single_step(self, batch, mode="train"):
        x, y = batch

        pred = self(x)
        loss = F.cross_entropy(pred, y)
        self.log(f"{mode}_loss", loss, prog_bar=True, logger=True)

        if mode == "val":
            y_hat = torch.argmax(pred, dim=1)
            accuracy = (y_hat == y).sum().item() / len(y)
            self.log("val_accuracy", accuracy, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, _):
        return self._single_step(batch, mode="train")

    def validation_step(self, batch, _):
        self._single_step(batch, mode="val")

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer

        scheduler_dict = {"scheduler": self.scheduler, "interval": "epoch", "monitor": "val_loss"}
        return [self.optimizer], [scheduler_dict]

    def on_fit_start(self):
        self.logger.log_hyperparams(
            {
                "flattened_dims": getattr(self.model, "flattened_dims", -1),
                "train_set_size": len(self.trainer.datamodule.train_dataset),
                "val_set_size": len(self.trainer.datamodule.val_dataset),
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

    model = Model(**configs)
    audio_datamodule = AudioDataModule(**configs)

    print("Model hparams")
    print(model.hparams)

    logger = WandbLogger(project=configs["project"])
    logger.experiment.watch(model, log="all")

    checkpoint_callback = WandbModelCheckpoint(
        dirpath=configs["checkpoints_dir"],
        filename=logger.experiment.name,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        every_n_val_epochs=1,
    )

    callbacks = [checkpoint_callback] + [LearningRateMonitor] if configs["scheduler"] else []

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger,
        callbacks=callbacks,
        max_epochs=configs["max_epochs"],
        check_val_every_n_epoch=configs["check_val_every_n_epoch"],
    )

    trainer.fit(model, datamodule=audio_datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--max_epochs", type=int, default=1_000)
    parser.add_argument("--project", type=str, default="TEST-hearingAId")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)

    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
