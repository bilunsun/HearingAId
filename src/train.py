import argparse
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision import models
from torch import nn

from lib import AudioDataModule, VanillaCNN, MobileNetV3Backbone, convnext_tiny, CustomCNN


class Scaler(nn.Module):
    '''Basic log scaler class'''

    def __init__(self, size: int):
        super().__init__()

        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("std", torch.ones(size))
        self.device = torch.device("cpu")

    def fit(self, x):
        x = torch.clip(x, min=1e-8)
        x = torch.log10(x).flatten()
        self.mean = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0)

    def transform(self, x):
        x = torch.clip(x, min=1e-8)
        return (torch.log10(x) - self.mean) / self.std

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.device = device

    def __repr__(self):
        return f'mean: {self.mean}\tstd: {self.std}'


class Model(pl.LightningModule):
    """
    Defines the hooks for training using pytorch lightning
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--width", type=int, default=128)
        parser.add_argument("--height", type=int, default=87)
        parser.add_argument("--n_classes", type=int, default=10)

        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--scheduler", type=str, choices=[None, "cosine", "plateau"], default=None)
        parser.add_argument("--min_lr", type=float, default=1e-7)
        parser.add_argument("--plateau_factor", type=float, default=0.1)
        parser.add_argument("--plateau_patience", type=int, default=10)
        parser.add_argument("--plateau_cooldown", type=int, default=0)
        parser.add_argument("--load_weights_from_ckpt", type=str, default=None)

        return parser

    def __init__(self, lr, scheduler, min_lr, plateau_factor, plateau_patience, plateau_cooldown, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        # self.model = MobileNetV3Backbone(n_classes=self.hparams.n_classes)
        # self.model = convnext_tiny(in_chans=1, num_classes=self.hparams.n_classes)
        # self.model = VanillaCNN(width=self.hparams.width, height=self.hparams.height, n_classes=self.hparams.n_classes)
        self.model = CustomCNN(n_classes=self.hparams.n_classes)
        # QUANTIZATION
        self.model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        # self.model = torch.quantization.fuse_modules(self.model, [["depthwise", "pointwise", "activation"]])
        self.model = torch.quantization.prepare_qat(self.model)

        if self.hparams.load_weights_from_ckpt:
            self.model = Model.load_from_checkpoint(self.hparams.load_weights_from_ckpt, load_weights_from_ckpt=None).model

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

        # self.scaler = Scaler((1, self.hparams.height, self.hparams.width))
        self.scaler = Scaler(size=())

        # For confusion matrix
        self._y_hat = []
        self._y = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        # x = (x - 17.2) / 252
        x = self.scaler.transform(x)

        pred = self(x)
        loss = F.cross_entropy(pred, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_mean", x.mean(), prog_bar=True, logger=True)
        self.log("train_std", x.std(), prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, _):
        x, y = batch
        # x = (x - 17.2) / 252
        x = self.scaler.transform(x)

        pred = self(x)
        loss = F.cross_entropy(pred, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        y_hat = torch.argmax(pred, dim=1)
        accuracy = (y == y_hat).float().mean()  # Need to cast Long tensor to Float
        self.log("val_accuracy", accuracy, prog_bar=True, logger=True)

        self._y_hat.append(y_hat)
        self._y.append(y)

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer

        scheduler_dict = {"scheduler": self.scheduler, "interval": "epoch", "monitor": "val_loss"}
        return [self.optimizer], [scheduler_dict]

    def on_fit_start(self):
        all_x, _ = self.trainer.datamodule.train_dataset[:]
        self.scaler.fit(all_x)
        self.scaler.to("cuda")
        print("mean.shape", self.scaler.mean.shape, "mean", self.scaler.mean)
        print("std.shape", self.scaler.std.shape, "std", self.scaler.std)

        self.class_names = self.trainer.datamodule.class_names

        self.logger.log_hyperparams(
            {
                "flattened_dims": getattr(self.model, "flattened_dims", -1),
                "train_set_size": len(self.trainer.datamodule.train_dataset),
                "val_set_size": len(self.trainer.datamodule.val_dataset),
                "class_names": self.class_names,
            }
        )

    def on_validation_epoch_end(self, *args) -> None:
        """
        Used to plot the confusion matrix
        """
        y = torch.cat(self._y).cpu().numpy()
        y_hat = torch.cat(self._y_hat).cpu().numpy()

        # accuracy = (y == y_hat).mean()  # Need to cast Long tensor to Float
        # self.log("val_accuracy", accuracy, prog_bar=True, logger=True)

        # import pdb; pdb.set_trace()
        self.logger.experiment.log(
            {
                "Confusion Matrix": wandb.plot.confusion_matrix(
                    y_true=y,
                    preds=y_hat,
                    class_names=self.class_names,
                )
            }
        )

        # Reset
        self._y = []
        self._y_hat = []


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

    audio_datamodule = AudioDataModule(**configs)
    configs["width"] = audio_datamodule.width
    configs["height"] = audio_datamodule.height
    configs["n_classes"] = audio_datamodule.n_classes
    model = Model(**configs)

    print("Model hparams")
    print(model.hparams)

    logger = WandbLogger(project=configs["project"])
    logger.experiment.watch(model, log="all")

    checkpoint_callback = ModelCheckpoint(
        dirpath=configs["checkpoints_dir"],
        filename=logger.experiment.name,
        save_top_k=1,
        monitor="val_accuracy",
        mode="max",
        every_n_epochs=1,
        verbose=True
    )

    callbacks = [checkpoint_callback]
    callbacks += [LearningRateMonitor()] if configs["scheduler"] else []

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
    parser.add_argument("--train_ratio", type=int, default=0.9)
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
