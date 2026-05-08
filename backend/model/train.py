import torch
from dataset import FoodDataset
from model import FoodCNN
from torch import nn
from torch._dynamo.decorators import F
from torch.utils.data import DataLoader

EPOCHS = 30
BATCH_SIZE = 32
LR = 0.001
DATA_DIR = "backend/trainingset/food-101"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = FoodDataset(DATA_DIR, split="train")
test_dataset = FoodDataset(DATA_DIR, split="test")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)

model = FoodCNN(num_classes=101).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    model.eval()
    test_loss = 0
    test_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            test_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_dataset)  # type: ignore
    test_acc = test_correct / len(test_dataset)  # type: ignore
    avg_train_loss = train_loss / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} "
        f"| Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f}"
    )

torch.save(model.state_dict(), "food_cnn.pth")
