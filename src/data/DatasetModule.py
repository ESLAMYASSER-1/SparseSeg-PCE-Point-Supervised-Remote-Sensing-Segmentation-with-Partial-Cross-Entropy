from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
import re
import os

def extract_city(filename):
    name = Path(filename).stem
    name = re.sub(r'_\d+$', '', name)
    return name

class OpenEarthMapDataset(Dataset):
    """
    Expected layout:
      root/
        <Area_name>/
          images/
          labels/
        

    OpenEarthMap label IDs in the public dataset are:
      0 Bareland, 1 Rangeland, 2 Developed space, 3 road, 4 Tree,
      5 Water, 6 Agriculture land, 7 Building.
    
    """
    def __init__(self, root, split="train", crop_size=512, transform=None):
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size
        self.transform = transform
        
        with open(f"{self.root}/{split}.txt", mode="+r") as f:
            images_names = f.read()
        
        images_names = images_names.split("\n")


        self.images = [
            f"{self.root}/{extract_city(image)}/images/{image}"

            for image in images_names

                if os.path.exists(
                    f"{self.root}/{extract_city(image)}/images/{image}"
                )
        ]
        self.masks = [
            f"{self.root}/{extract_city(image)}/labels/{image}"

            for image in images_names
            
                if os.path.exists(
                    f"{self.root}/{extract_city(image)}/labels/{image}"
                )
        ]


        if not self.images:
            raise RuntimeError(f"No PNG images found. ")

    def __len__(self):
        return len(self.images)

    def _crop(self, image, mask):
        h, w = mask.shape
        cs = min(self.crop_size, h, w)
        if h == cs:
            top = 0
        else:
            top = np.random.randint(0, h-cs+1)
        if w == cs:
            left = 0
        else:
            left = np.random.randint(0, w-cs+1)
        return image[top:top+cs, left:left+cs], mask[top:top+cs, left:left+cs]

    def __getitem__(self, idx):
        ip = self.images[idx]
        mp = self.masks[idx]

        image = np.array(Image.open(ip).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mp), dtype=np.int64)

        image, mask = self._crop(image, mask)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        return image, mask
