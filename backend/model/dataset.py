import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as transforms


class FoodDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = Path(root_dir)

        with open(self.root_dir / "meta" / f"{split}.json") as f:
            split_data = json.load(f)

        self.classes = sorted(split_data.keys())

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for cls, images in split_data.items():
            label = self.class_to_idx[cls]
            for img_id in images:
                img_path = self.root_dir / "images" / f"{img_id}.jpg"
                self.samples.append((img_path, label))

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        def len(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, label
