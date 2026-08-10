import torch
import lightning as L
import segmentation_models_pytorch as smp
from utils import PCELoss, segmentation_metrics, make_point_labels

class UNetLightning(L.LightningModule):

    def __init__(
        self,
        in_channels=3,
        num_classes=9,
        fraction=0.3, 
        ignore_index=-1,
        learning_rate=1e-3,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.num_classes = num_classes
        self.fraction = fraction
        self.ignore_index = ignore_index


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
        masks = make_point_labels(masks, self.fraction, self.ignore_index)

        logits = self(images)
        pred = logits.argmax(dim=1).cpu()

        loss = self.loss_fn(logits, masks)
        metrics = segmentation_metrics(masks.cpu(), pred, self.num_classes, "train")

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True
        )

        per_class_iou= metrics["train_per_class_iou"]
        for i, iou in enumerate(per_class_iou):
            self.log(
                f"train/iou_class_{i}",
                iou,
                on_step=False,
                on_epoch=True,
            )
            
        metrics.pop("train_per_class_iou")

        self.log_dict(
                        metrics,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch

        logits = self(images)
        pred = logits.argmax(dim=1).cpu()

        loss = self.loss_fn(logits, masks)
        
        metrics = segmentation_metrics(masks.cpu(), pred, self.num_classes, "val")
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True
        )

        per_class_iou= metrics["val_per_class_iou"]
        for i, iou in enumerate(per_class_iou):
            self.log(
                f"val/iou_class_{i}",
                iou,
                on_step=False,
                on_epoch=True,
            )

        metrics.pop("val_per_class_iou")
        self.log_dict(
                        metrics,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
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