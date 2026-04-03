"""
Dental Tooth Classifier — Training Script
Model  : EfficientNet-B0 (ImageNet pretrained, fine-tuned)
Data   : USE CASE - 01 / Segmented Dental Radiography  (train / valid / test)
Classes: Cavity | Fillings | Impacted Tooth | Implant | Normal
Output : models/efficientnet_dental.pth
"""

import os
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
DATA_ROOT = Path("E:/Downloads/Data/USE CASE - 01/Segmented Dental Radiography")
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "efficientnet_dental.pth"

# ── Hyper-parameters ───────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Cavity", "Fillings", "Impacted Tooth", "Implant", "Normal"]


# ── Transforms ─────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_weighted_sampler(dataset):
    """Balance minority classes (Cavity, Impacted Tooth) during training."""
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def build_model(num_classes: int):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def train():
    print(f"\n{'='*55}")
    print(f"  Dental Classifier Training")
    print(f"  Device : {DEVICE}")
    print(f"  Data   : {DATA_ROOT}")
    print(f"  Epochs : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LR}")
    print(f"{'='*55}\n")

    # ── Datasets ───────────────────────────────────────────────────────────────
    train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=train_tf)
    valid_ds = datasets.ImageFolder(DATA_ROOT / "valid", transform=val_tf)
    test_ds  = datasets.ImageFolder(DATA_ROOT / "test",  transform=val_tf)

    print(f"Classes  : {train_ds.classes}")
    print(f"Train    : {len(train_ds):,}  |  Valid : {len(valid_ds):,}  |  Test : {len(test_ds):,}\n")

    sampler    = get_weighted_sampler(train_ds)
    train_dl   = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                            num_workers=2, pin_memory=True)
    valid_dl   = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_dl    = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    model     = build_model(len(train_ds.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_acc   = 0.0
    best_state = None

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss, train_correct = 0.0, 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()

        scheduler.step()

        # Validate
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in valid_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out  = model(imgs)
                loss = criterion(out, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_ds)
        val_acc   = val_correct   / len(valid_ds)
        elapsed   = time.time() - t0

        marker = " << BEST" if val_acc > best_acc else ""
        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss/len(train_ds):.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss/len(valid_ds):.4f}  val_acc={val_acc:.4f}  "
              f"[{elapsed:.0f}s]{marker}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # ── Save best model ────────────────────────────────────────────────────────
    torch.save({
        "model_state":  best_state,
        "class_names":  train_ds.classes,
        "img_size":     IMG_SIZE,
        "architecture": "efficientnet_b0",
    }, MODEL_PATH)

    print(f"\nBest val accuracy : {best_acc:.4f}")
    print(f"Model saved       : {MODEL_PATH}\n")

    # ── Test set evaluation ────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    test_correct = 0
    class_correct = [0] * len(train_ds.classes)
    class_total   = [0] * len(train_ds.classes)

    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            preds = out.argmax(1)
            test_correct += (preds == labels).sum().item()
            for p, l in zip(preds, labels):
                class_correct[l] += int(p == l)
                class_total[l]   += 1

    print(f"Test Accuracy : {test_correct/len(test_ds):.4f}")
    print("\nPer-class accuracy:")
    for i, cls in enumerate(train_ds.classes):
        acc = class_correct[i] / class_total[i] if class_total[i] else 0
        print(f"  {cls:<18} {acc:.4f}  ({class_correct[i]}/{class_total[i]})")

    print(f"\nDone! Use 'python main.py' to start the server with the local model.")


if __name__ == "__main__":
    train()
