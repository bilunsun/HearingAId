import argparse
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from tqdm.auto import tqdm

from lib import AudioDataModule, EnvNet, StandardScaler, MelScaler
from pretrain_mix import PretrainMix
from pretrain_simsiam import PretrainSimSiam


class Model(pl.LightningModule):
    """
    Defines the hooks for training using pytorch lightning
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--width", type=int, default=None)
        parser.add_argument("--height", type=int, default=None)
        parser.add_argument("--classifier_hidden_dims", type=int, default=512)
        parser.add_argument("--n_classes", type=int, default=10)

        parser.add_argument("--transfer_ckpt", type=str, default=None)
        parser.add_argument("--mix_ckpt", type=str, default=None)
        parser.add_argument("--simsiam_ckpt", type=str, default=None)

        parser.add_argument("--qat", action="store_true")
        parser.add_argument("--lr", type=float, default=5e-4)

        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.scaler = MelScaler(size=()) if self.hparams.convert_to_mel else StandardScaler(size=())

        if self.hparams.transfer_ckpt is not None:
            print("Transferring from", self.hparams.transfer_ckpt)

            ckpt = Model.load_from_checkpoint(self.hparams.transfer_ckpt, transfer_ckpt=None)

            # Set to none; otherwise when loading the current model,
            # the original transfer_ckpt will be needed
            self.hparams.transfer_ckpt = None

            self.scaler = ckpt.scaler
            self.model = ckpt.model

            if self.model.classifier[-1].out_features != self.hparams.n_classes:
                self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, self.hparams.n_classes)

            for p in self.model.parameters():
                p.requires_grad_(False)

            for p in self.model.classifier:
                p.requires_grad_(True)

        elif self.hparams.mix_ckpt is not None:
            ckpt = PretrainMix.load_from_checkpoint(self.hparams.mix_ckpt, width=self.hparams.width, height=self.hparams.height)
            self.scaler = ckpt.scaler
            self.model = ckpt.backbone

            self.model.classifier = nn.Sequential(
                nn.Linear(self.model.classifier[-1].in_features, self.hparams.classifier_hidden_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.hparams.classifier_hidden_dims, self.hparams.n_classes)
            )

        elif self.hparams.simsiam_ckpt is not None:
            ckpt = PretrainSimSiam.load_from_checkpoint(self.hparams.simsiam_ckpt, strict=False)
            self.scaler = ckpt.scaler

            # SimSiam pretrained models need a classifier
            self.model = nn.Sequential(
                ckpt.backbone,
                nn.Sequential(
                    nn.Linear(ckpt.latent_dim, 1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, self.hparams.n_classes)
                )
            )
        else:
            self.model = EnvNet(n_classes=self.hparams.n_classes, height=self.hparams.height, width=self.hparams.width, classifier_hidden_dims=self.hparams.classifier_hidden_dims)

        # QUANTIZATION
        if self.hparams.qat:
            self.model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
            self.model = torch.quantization.prepare_qat(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        # For confusion matrix
        self._y_hat = []
        self._y = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        x = self.scaler.transform(x)

        pred = self(x)
        loss = F.cross_entropy(pred, y)

        y_hat = torch.argmax(pred, dim=1)
        accuracy = (y == y_hat).float().mean()

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_mean", x.mean(), prog_bar=True, logger=True)
        self.log("train_std", x.std(), prog_bar=True, logger=True)
        self.log("train_accuracy", accuracy, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, _):
        x, y = batch
        x = self.scaler.transform(x)

        pred = self(x)
        loss = F.cross_entropy(pred, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        y_hat = torch.argmax(pred, dim=1)

        self._y_hat.append(y_hat)
        self._y.append(y)

    def configure_optimizers(self):
        return self.optimizer

    def on_fit_start(self):
        if not (self.hparams.mix_ckpt or self.hparams.simsiam_ckpt):
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
                "class_names": self.class_names,
            }
        )

    def on_validation_epoch_end(self, *args) -> None:
        """
        Used to log the mean accuracy over the entire validation set
        """
        y = torch.cat(self._y).cpu().numpy()
        y_hat = torch.cat(self._y_hat).cpu().numpy()

        accuracy = (y == y_hat).mean()  # Need to cast Long tensor to Float
        self.log("val_accuracy", accuracy, prog_bar=True, logger=True)

        # Reset
        self._y = []
        self._y_hat = []

    def on_save_checkpoint(self, checkpoint):
        checkpoint["class_names"] = self.class_names

    def on_load_checkpoint(self, checkpoint):
        self.class_names = checkpoint.get("class_names")


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
        verbose=True,
    )

    callbacks = [checkpoint_callback]

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger,
        callbacks=callbacks,
        max_epochs=configs["max_epochs"],
        check_val_every_n_epoch=configs["check_val_every_n_epoch"],
        stochastic_weight_avg=True
    )

    trainer.fit(model, datamodule=audio_datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="urbansound8k")
    parser.add_argument("--target_sample_rate", type=int, default=16_000)
    parser.add_argument("--n_samples", type=int, default=64_000)
    parser.add_argument("--convert_to_mel", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--max_epochs", type=int, default=5_000)
    parser.add_argument("--project", type=str, default="TEST-hearingAId")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)

    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
