import copy
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights


DATASET_ROOT = "/data"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "validation")

CHECKPOINT_DIR = "/app/models"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "mobilenetv3_small_7classes_best.pth",
)

CLASS_NAMES = [
    "Bougainvillea",
    "Crown of thorns",
    "Hibiscus",
    "Jungle geranium",
    "Madagascar periwinkle",
    "Marigold",
    "Rose",
]

NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224
BATCH_SIZE = 32

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 20

PHASE1_LR = 1e-3
BACKBONE_LR = 1e-4
CLASSIFIER_LR = 1e-3

WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

NUM_WORKERS = 2

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")

weights = MobileNet_V3_Small_Weights.DEFAULT

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform,
)

expected_mapping = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

if train_dataset.class_to_idx != expected_mapping:
    raise RuntimeError(
        "Le mapping des classes du dataset train ne correspond "
        "pas au mapping attendu."
    )

if val_dataset.class_to_idx != expected_mapping:
    raise RuntimeError(
        "Le mapping des classes du dataset validation ne correspond "
        "pas au mapping attendu."
    )

print("Class mapping:")
for class_name, index in expected_mapping.items():
    print(f"  {index}: {class_name}")

print(f"Train images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

model = models.mobilenet_v3_small(
    weights=weights,
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    NUM_CLASSES,
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()


def calculate_accuracy(outputs, targets):
    predictions = outputs.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    return correct


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)

        running_loss += loss.item() * batch_size
        correct += calculate_accuracy(outputs, targets)
        total += batch_size

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)

        running_loss += loss.item() * batch_size
        correct += calculate_accuracy(outputs, targets)
        total += batch_size

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def save_checkpoint(
    model,
    epoch,
    phase,
    val_loss,
    val_accuracy,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "class_names": CLASS_NAMES,
        "class_to_idx": expected_mapping,
        "epoch": epoch,
        "phase": phase,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "model_name": "MobileNetV3-Small",
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
    }

    torch.save(checkpoint, CHECKPOINT_PATH)

    print(f"Checkpoint saved: {CHECKPOINT_PATH}")


print("\n" + "=" * 60)
print("PHASE 1 — CLASSIFIEUR")
print("=" * 60)

for parameter in model.parameters():
    parameter.requires_grad = False

for parameter in model.classifier[3].parameters():
    parameter.requires_grad = True

optimizer = torch.optim.AdamW(
    model.classifier[3].parameters(),
    lr=PHASE1_LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=PHASE1_EPOCHS,
)

best_val_loss = float("inf")
best_val_accuracy = 0.0
best_epoch = 0
best_model_state = copy.deepcopy(model.state_dict())

for epoch in range(1, PHASE1_EPOCHS + 1):
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
    )

    val_loss, val_accuracy = validate(
        model,
        val_loader,
        criterion,
    )

    scheduler.step()

    print(
        f"Phase 1 | Epoch {epoch}/{PHASE1_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_accuracy = val_accuracy
        best_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())

        save_checkpoint(
            model=model,
            epoch=epoch,
            phase=1,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
        )

model.load_state_dict(best_model_state)

print("\n" + "=" * 60)
print("PHASE 2 — FINE-TUNING")
print("=" * 60)

for parameter in model.parameters():
    parameter.requires_grad = True

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.features.parameters(),
            "lr": BACKBONE_LR,
        },
        {
            "params": model.classifier.parameters(),
            "lr": CLASSIFIER_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=PHASE2_EPOCHS,
)

epochs_without_improvement = 0

for epoch in range(1, PHASE2_EPOCHS + 1):
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
    )

    val_loss, val_accuracy = validate(
        model,
        val_loader,
        criterion,
    )

    scheduler.step()

    print(
        f"Phase 2 | Epoch {epoch}/{PHASE2_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_accuracy = val_accuracy
        best_epoch = epoch

        epochs_without_improvement = 0

        save_checkpoint(
            model=model,
            epoch=epoch,
            phase=2,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
        )

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(
            f"Early stopping après "
            f"{EARLY_STOPPING_PATIENCE} epochs sans amélioration."
        )
        break

print("\n" + "=" * 60)
print("ENTRAÎNEMENT TERMINÉ")
print("=" * 60)

print(f"Meilleure epoch : {best_epoch}")
print(f"Meilleure val_loss : {best_val_loss:.4f}")
print(f"Val accuracy correspondante : {best_val_accuracy:.4f}")
print(f"Checkpoint : {CHECKPOINT_PATH}")
