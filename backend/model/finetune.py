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
STAGE1_EPOCHS = 8
STAGE2_EPOCHS = 25
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
    device: torch.device,  # type: ignore
    training: bool,
) -> tuple[float, float]:
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            if training and optimizer:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if training and optimizer:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def freeze_conv_blocks(model: FoodCNN):
    for name, param in model.named_parameters():
        if name.startswith("fc"):
            param.requires_grad = True

        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 1: {trainable:,} trainable parameters (FC layers only)")


def unfreeze_all(model: FoodCNN):
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 2: {trainable:,} trainable parameters (all layers)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    print(f"Device: {device}")

    with open(ALL_CLASSES_JSON) as f:
        all_classes = sorted(json.load(f).keys())

    train_samples, val_samples = build_samples(all_classes)
    print(f"Train: {len(train_samples)} images, Val: {len(val_samples)} images")

    train_ds = CropsDataset(train_samples, train_transform)
    val_ds = CropsDataset(val_samples, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = FoodCNN(num_classes=101).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {CHECKPOINT}")

    criterion = nn.CrossEntropyLoss()

    print("\n--- Stage 1: FC layers only ---")
    freeze_conv_blocks(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE1_LR,
        weight_decay=1e-4,
    )

    best_val_acc = 0.0

    for epoch in range(1, STAGE1_EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, training=False
        )

        print(
            f"Epoch {epoch}/{STAGE1_EPOCHS} "
            f"| train loss: {train_loss:.4f} acc: {train_acc:.3f} "
            f"| val loss: {val_loss:.4f} acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT_STAGE1)
            print(f" saved stage 1 checkpoint (val acc {val_acc:.3f})")

    print(f"\nStage 1 done. Best val acc: {best_val_acc:.3f}")

    model.load_state_dict(
        torch.load(OUT_STAGE1, map_location=device, weights_only=True)
    )

    print("\n--- Stage 2: all layers ---")
    unfreeze_all(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=STAGE2_LR, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STAGE2_EPOCHS, eta_min=1e-6
    )

    best_val_acc = 0.0

    for epoch in range(1, STAGE2_EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, training=False
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch}/{STAGE2_EPOCHS} "
            f"| train loss: {train_loss:.4f} acc: {train_acc:.3f}"
            f"| val loss: {val_loss:.4f} acc: {val_acc:.3f}"
            f"| lr: {current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT_STAGE2)
            print(f" Saved stage 2 checkpoint (val acc {val_acc:.3f})")

    print(f"\nStage 2 done. Best val acc: {best_val_acc:.3f}")
    print(f"Final checkpoint: {OUT_STAGE2}")
    print("\nNext: run export.py pointing at foodcnn_finetune_stage2.pth")


if __name__ == "__main__":
    main()
