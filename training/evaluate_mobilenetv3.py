import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights


DATASET_ROOT = "/workspace/datasets/tropical_flower/split_clean"
TEST_DIR = os.path.join(DATASET_ROOT, "test")

CHECKPOINT_PATH = "/workspace/models/mobilenetv3_small_7classes_best.pth"

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
NUM_WORKERS = 2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

if checkpoint.get("model_name") != "MobileNetV3-Small":
    raise RuntimeError(
        f"Modèle inattendu dans le checkpoint : "
        f"{checkpoint.get('model_name')}"
    )

if checkpoint.get("num_classes") != NUM_CLASSES:
    raise RuntimeError(
        f"Nombre de classes inattendu : "
        f"{checkpoint.get('num_classes')} != {NUM_CLASSES}"
    )

if checkpoint.get("image_size") != IMAGE_SIZE:
    raise RuntimeError(
        f"Taille d'image inattendue : "
        f"{checkpoint.get('image_size')} != {IMAGE_SIZE}"
    )

if checkpoint.get("class_names") != CLASS_NAMES:
    raise RuntimeError(
        "Le mapping des classes du checkpoint ne correspond pas "
        "au mapping attendu."
    )


test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform,
)

expected_mapping = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

if test_dataset.class_to_idx != expected_mapping:
    raise RuntimeError(
        "Le mapping des classes du dataset test ne correspond pas "
        "au mapping attendu."
    )


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


model = models.mobilenet_v3_small(
    weights=MobileNet_V3_Small_Weights.DEFAULT
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    NUM_CLASSES,
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()


criterion = nn.CrossEntropyLoss()


total_loss = 0.0
total_correct = 0
total_samples = 0

correct_per_class = [0] * NUM_CLASSES
total_per_class = [0] * NUM_CLASSES

confusion_matrix = torch.zeros(
    NUM_CLASSES,
    NUM_CLASSES,
    dtype=torch.int64,
)


with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, targets)

        predictions = outputs.argmax(dim=1)

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == targets).sum().item()
        total_samples += batch_size

        for target, prediction in zip(targets, predictions):
            target_index = target.item()
            prediction_index = prediction.item()

            total_per_class[target_index] += 1

            if target_index == prediction_index:
                correct_per_class[target_index] += 1

            confusion_matrix[target_index, prediction_index] += 1


global_loss = total_loss / total_samples
global_accuracy = total_correct / total_samples


print()
print("=" * 70)
print("GREENCheck AI — ÉVALUATION FINALE DU MODÈLE")
print("=" * 70)
print()
print(f"Modèle             : {checkpoint['model_name']}")
print(f"Checkpoint         : {CHECKPOINT_PATH}")
print(f"Epoch sélectionné  : {checkpoint['epoch']}")
print(f"Dataset             : {TEST_DIR}")
print(f"Nombre d'images    : {total_samples}")
print(f"Nombre de classes  : {NUM_CLASSES}")
print(f"Device             : {DEVICE}")
print()
print(f"Test loss          : {global_loss:.6f}")
print(f"Test accuracy      : {global_accuracy:.4%}")
print()
print("-" * 70)
print("ACCURACY PAR CLASSE")
print("-" * 70)

for index, class_name in enumerate(CLASS_NAMES):
    total = total_per_class[index]
    correct = correct_per_class[index]
    incorrect = total - correct
    accuracy = correct / total if total else 0.0

    print(
        f"{index} → {class_name}: "
        f"{accuracy:.4%} "
        f"({correct}/{total} correct, {incorrect} incorrect)"
    )

print()
print("-" * 70)
print("MATRICE DE CONFUSION")
print("-" * 70)

print()
print("Lignes = classe réelle")
print("Colonnes = classe prédite")
print()

header = " " * 26 + " ".join(
    f"{index:>8}"
    for index in range(NUM_CLASSES)
)

print(header)

for index, class_name in enumerate(CLASS_NAMES):
    row = " ".join(
        f"{confusion_matrix[index, column].item():>8}"
        for column in range(NUM_CLASSES)
    )

    print(f"{index} {class_name:<22} {row}")

print()
print("Légende :")
for index, class_name in enumerate(CLASS_NAMES):
    print(f"{index} = {class_name}")

print()
print("=" * 70)
