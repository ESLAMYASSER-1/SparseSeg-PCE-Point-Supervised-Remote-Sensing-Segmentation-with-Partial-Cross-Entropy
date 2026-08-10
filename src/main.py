from data import OpenEarthMapDataset
from models import UNetLightning
from utils import check_folders

import random
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader


check_folders("./OpenEarthMap/OpenEarthMap_wo_xBD")
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

dmT = OpenEarthMapDataset("./OpenEarthMap/OpenEarthMap_wo_xBD", "train", crop_size=224)
dmV = OpenEarthMapDataset("./OpenEarthMap/OpenEarthMap_wo_xBD", "val", crop_size=224)
dmTest = OpenEarthMapDataset("./OpenEarthMap/OpenEarthMap_wo_xBD", "test", crop_size=224)

train_loader = DataLoader(
    dmT,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    dmV,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    dmTest,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

fractions = [0.01, 0.05, 0.1, 0.2]
seeds = [42, 43, 44]
for seed in seeds:
    for frac in fractions:
        early_stopping = EarlyStopping(
            monitor="val_mIoU",
            mode="max",
            patience=10,
            min_delta=0.001,
            verbose=True,
        )

        checkpoint = ModelCheckpoint(
            monitor="val_mIoU",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_mIoU:.4f}",
        )

        model = UNetLightning(
            in_channels=3,
            num_classes=9,
            learning_rate=2e-3,
            fraction=frac, 
            ignore_index=-1,
        )

        trainer = L.Trainer(
            max_epochs=100,
            accelerator="auto",
            devices="auto",
            precision="16-mixed",
            callbacks=[
                early_stopping,
                checkpoint,
            ],
        )

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )