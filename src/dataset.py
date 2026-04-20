import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, split='train', transform=None):
        """
        Args:
            root_dir: path to mvtec dataset
            category: e.g. 'bottle', 'cable' etc.
            split: 'train' or 'test'
            transform: torchvision transforms
        """
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []  # 0 = normal, 1 = defect

        self._load_dataset()

    def _load_dataset(self):
        split_dir = self.root_dir / self.category / self.split

        for label_dir in sorted(split_dir.iterdir()):
            label = 0 if label_dir.name == 'good' else 1
            for img_path in sorted(label_dir.glob('*.png')):
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label, str(self.images[idx])


def get_transforms(split='train', img_size=224):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])