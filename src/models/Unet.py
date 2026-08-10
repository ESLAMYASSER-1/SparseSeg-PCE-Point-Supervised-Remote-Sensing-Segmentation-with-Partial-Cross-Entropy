import torch
import lightning as L
import segmentation_models_pytorch as smp
from utils import PCELoss

class UNetLightning(L.LightningModule):

    def __init__(
        self,
        in_channels=3,
        num_classes=9,
        learning_rate=1e-3
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )

        self.loss_fn = PCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch

        logits = self(images)

        loss = self.loss_fn(logits, masks)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch

        logits = self(images)

        loss = self.loss_fn(logits, masks)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True
        )

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }