import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class ClothingDataset(Dataset):
    def __init__(self, df, image_dir, class_names, transform=None):
        self.df          = df.reset_index(drop=True)
        self.image_dir   = image_dir
        self.class_names = class_names
        self.transform   = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        image  = Image.open(os.path.join(self.image_dir, row['image_path'])).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.class_names].values.astype(float), dtype=torch.float32)
        return image, labels
