import json
from pathlib import Path

import torch
import torch.nn as nn
from model import FoodCNN
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).parent
CROPS_DIR = ROOT / "trainingset" / "food-101-cropped" / "images"
CHECKPOINT = ROOT / "checkpoints" / "foodcnn_best.pth"
OUT_STAGE1 = ROOT / "checkpoints" / "foodcnn_finetune_stage1.pth"
OUT_STAGE2 = ROOT / "checkpoints" / "foodcnn_finetune_stage2.pth"

SUBSET_CLASSES = [
    "pizza",
    "sushi",
    "ramen",
    "hamburger",
    "steak",
    "ice_cream",
    "waffles",
    "tacos",
    "fried_rice",
    "caesar_salad",
]

ALL_CLASSES_JSON = ROOT / "trainingset" / "food-101" / "meta" / "train.json"

BATCH_SIZE = 32
STAGE1_LR = 0.001
STAGE2_LR = 0.0001
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 15
VAL_SPLIT = 0.15

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CropsDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_samples(all_classes: list[str]) -> tuple[list, list]:
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

    all_samples = []
    for cls in SUBSET_CLASSES:
        cls_dir = CROPS_DIR / cls
        if not cls_dir.exists():
            print(f"Warning: {cls_dir} not found, skipping...")
            continue

        idx = class_to_idx[cls]
        paths = sorted(cls_dir.glob("*.jpg"))

        for p in paths:
            all_samples.append((p, idx))

    import random

    random.seed(42)
    random.shuffle(all_samples)

    split = int(len(all_samples) * (1 - VAL_SPLIT))
    return all_samples[:split], all_samples[split:]


train_transform = transforms.Compose(
    [
        transforms.Resize((144, 144)),
        transforms.RandomCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# Val transform — deterministic, no augmentation
val_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    training: bool,
) -> tuple[float, float]:
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
