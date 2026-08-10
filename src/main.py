from data import OpenEarthMapDataset
from models import UNetLightning
from utils import check_folders

import torch
import lightning as L
from torch.utils.data import DataLoader


check_folders("./OpenEarthMap/OpenEarthMap_wo_xBD")


dmT = OpenEarthMapDataset("./OpenEarthMap/OpenEarthMap_wo_xBD", "train", crop_size=224)
dmV = OpenEarthMapDataset("./OpenEarthMap/OpenEarthMap_wo_xBD", "val", crop_size=224)
dmTest = OpenEarthMapDataset("./OpenEarthMap/OpenEarthMap_wo_xBD", "test", crop_size=224)

train_loader = DataLoader(
    dmT,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# images, masks = next(iter(train_loader))

# print("shape:", masks.shape)
# print("dtype:", masks.dtype)
# print("unique:", torch.unique(masks))
# print("min:", masks.min())
# print("max:", masks.max())

val_loader = DataLoader(
    dmV,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    dmTest,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


model = UNetLightning(
    in_channels=3,
    num_classes=9,
    learning_rate=1e-3
)

trainer = L.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices="auto",
    precision="16-mixed"
)

trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader
)