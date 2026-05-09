import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from dataset import FoodDataset
from model import FoodCNN
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
print(f"Using device: {device}")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.abspath("checkpoints/foodcnn.pth")
    best_checkpoint_path = os.path.abspath("checkpoints/foodcnn_best.pth")

    model = FoodCNN(num_classes=101).to(device)

    train_dataset = FoodDataset("trainingset/food-101", split="train")
    test_dataset = FoodDataset("trainingset/food-101", split="test")
    print(f"Train: {len(train_dataset)} images | Test: {len(test_dataset)} images")

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")  # type: ignore
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=45, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Resumed from existing checkpoint.")

    best_loss = float("inf")

    for epoch in range(50):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):  # type: ignore
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        model.eval()
        top1_correct = 0
        top5_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                top1_correct += (outputs.argmax(1) == labels).sum().item()

                top5_indices = torch.topk(outputs, 5, dim=1).indices  # type: ignore
                top5_correct += sum(
                    labels[i].item() in top5_indices[i].tolist()
                    for i in range(len(labels))
                )

        top1_acc = top1_correct / len(test_dataset)
        top5_acc = top5_correct / len(test_dataset)
        print(
            f"Epoch {epoch + 1:03d} | loss: {avg_loss:.4f} | top1: {top1_acc:.4f} | top5: {top5_acc:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  ★ New best ({best_loss:.4f}) saved to {best_checkpoint_path}")

    torch.save(model.state_dict(), checkpoint_path)
    print("\nTraining complete.")
    print(f"  Final : {checkpoint_path}")
    print(f"  Best  : {best_checkpoint_path}  (loss {best_loss:.4f})")
