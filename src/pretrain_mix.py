import argparse
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from tqdm.auto import tqdm

from lib import AudioDataModule, EnvNet, StandardScaler


class PretrainMix(pl.LightningModule):
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

        self.backbone = EnvNet(width=self.hparams.width, height=self.hparams.height, n_classes=self.hparams.n_classes)

        self.optimizer = torch.optim.Adam(self.backbone.parameters(), lr=self.hparams.lr)
        self.criterion = nn.KLDivLoss(reduction="batchmean")

        self.scaler = StandardScaler(size=())

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, _):
        x, y = batch
        x = self.scaler.transform(x)

        pred = F.log_softmax(self(x), dim=1)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_mean", x.mean(), prog_bar=True, logger=True)
        self.log("train_std", x.std(), prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, _):
        x, y = batch
        x = self.scaler.transform(x)

        pred = F.log_softmax(self(x), dim=1)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_mean", x.mean(), prog_bar=True, logger=True)
        self.log("val_std", x.std(), prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return self.optimizer

    def on_fit_start(self):
        all_x = []
        for i, (x, _) in enumerate(tqdm(self.trainer.datamodule.train_dataloader())):
            if i == 10:
                break
            all_x.append(x)
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


def main(args):
    configs = vars(args)

    # Set the seed; pl.seed_everything sets a random seed if args.seed is None
    seed = pl.seed_everything(configs["seed"])
    configs["seed"] = seed

    # Pretrain
    configs["pretraining"] = "mix"

    audio_datamodule = AudioDataModule(**configs)
    configs["width"] = audio_datamodule.width
    configs["height"] = audio_datamodule.height
    configs["n_classes"] = audio_datamodule.n_classes
    model = PretrainMix(**configs)

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

    trainer.fit(model, datamodule=audio_datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="urbansound8k")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--max_epochs", type=int, default=5_000)
    parser.add_argument("--project", type=str, default="TEST-hearingAId")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)

    parser = PretrainMix.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
