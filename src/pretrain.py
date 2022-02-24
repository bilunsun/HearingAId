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


# class Scaler(nn.Module):
#     """Basic log scaler class"""

#     def __init__(self, size: int):
#         super().__init__()

#         self.register_buffer("mean", torch.zeros(size))
#         self.register_buffer("std", torch.ones(size))
#         self.device = torch.device("cpu")

#     def fit(self, x):
#         x = torch.clip(x, min=1e-8)
#         x = torch.log10(x).flatten()
#         self.mean = torch.mean(x, dim=0)
#         self.std = torch.std(x, dim=0)

#     def transform(self, x):
#         x = torch.clip(x, min=1e-8)
#         return (torch.log10(x) - self.mean) / self.std

#     def to(self, device):
#         self.mean = self.mean.to(device)
#         self.std = self.std.to(device)
#         self.device = device

#     def __repr__(self):
#         return f"mean: {self.mean}\tstd: {self.std}"


class Scaler(nn.Module):
    """Basic log scaler class"""

    def __init__(self, size: int):
        super().__init__()

        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("std", torch.ones(size) / 10)
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


class PretrainModel(pl.LightningModule):
    """
    Defines the hooks for training using pytorch lightning
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--n_classes", type=int, default=10)
        parser.add_argument("--transfer_ckpt", type=str, default=None)
        parser.add_argument("--lr", type=float, default=5e-4)

        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.model = EnvNet(n_classes=self.hparams.n_classes)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        self.criterion = nn.KLDivLoss(reduction="batchmean")

        self.scaler = Scaler(size=())

        # self._y_hat = []
        # self._y = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        x = self.scaler.transform(x)

        pred = F.log_softmax(self(x), dim=1)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_mean", x.mean(), prog_bar=True, logger=True)
        self.log("train_std", x.std(), prog_bar=True, logger=True)

        return loss

    # def validation_step(self, batch, _):
    #     x, y = batch
    #     x = self.scaler.transform(x)

    #     pred = self(x)
    #     loss = self.criterion(pred, y)
    #     self.log("val_loss", loss, prog_bar=True, logger=True)

    #     y_hat = torch.argmax(pred, dim=1)

    #     self._y_hat.append(y_hat)
    #     self._y.append(y)

    def configure_optimizers(self):
        return self.optimizer

    def on_fit_start(self):
        # all_x = []
        # for x, _ in tqdm(self.trainer.datamodule.train_dataloader()):
        #     all_x.append(x)
        # all_x = torch.cat(all_x)

        # self.scaler.fit(all_x)
        self.scaler.to("cuda")
        print("mean.shape", self.scaler.mean.shape, "mean", self.scaler.mean)
        print("std.shape", self.scaler.std.shape, "std", self.scaler.std)

        self.class_names = self.trainer.datamodule.class_names

        self.logger.log_hyperparams(
            {
                "train_set_size": len(self.trainer.datamodule.train_dataset),
                "val_set_size": len(self.trainer.datamodule.val_dataset),
                "class_names": self.class_names,
            }
        )

    # def on_validation_epoch_end(self, *args) -> None:
    #     """
    #     Used to plot the confusion matrix
    #     """
    #     y = torch.cat(self._y).cpu().numpy()
    #     y_hat = torch.cat(self._y_hat).cpu().numpy()

    #     accuracy = (y == y_hat).mean()  # Need to cast Long tensor to Float
    #     self.log("val_accuracy", accuracy, prog_bar=True, logger=True)

    #     # # import pdb; pdb.set_trace()
    #     # self.logger.experiment.log(
    #     #     {"Confusion Matrix": wandb.plot.confusion_matrix(y_true=y, preds=y_hat, class_names=self.class_names,)}
    #     # )

    #     # Reset
    #     self._y = []
    #     self._y_hat = []


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
    configs["pretraining"] = True

    audio_datamodule = AudioDataModule(**configs)
    configs["width"] = audio_datamodule.width
    configs["height"] = audio_datamodule.height
    configs["n_classes"] = audio_datamodule.n_classes
    model = PretrainModel(**configs)

    print("Model hparams")
    print(model.hparams)

    logger = WandbLogger(project=configs["project"])
    logger.experiment.watch(model, log="all")

    checkpoint_callback = ModelCheckpoint(
        dirpath=configs["checkpoints_dir"],
        filename=logger.experiment.name,
        save_top_k=5,
        monitor="train_loss",
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

    trainer.fit(model, datamodule=audio_datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="urbansound8k")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_ratio", type=int, default=0.99)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--max_epochs", type=int, default=1_000)
    parser.add_argument("--project", type=str, default="TEST-hearingAId")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)

    parser = PretrainModel.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
