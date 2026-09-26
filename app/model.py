import os

import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


class PlantModel:
    """
    Gestion du modèle MobileNetV3-Small spécialisé GreenCheck.
    """

    def __init__(self):
        print("Chargement du modèle GreenCheck AI...")

        self.device = torch.device("cpu")

        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models",
            "mobilenetv3_small_7classes_best.pth",
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint introuvable : {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.class_names = checkpoint["class_names"]
        self.class_to_idx = checkpoint["class_to_idx"]
        self.model_name = checkpoint["model_name"]
        self.num_classes = checkpoint["num_classes"]
        self.image_size = checkpoint["image_size"]

        weights = MobileNet_V3_Small_Weights.DEFAULT

        self.model = mobilenet_v3_small(
            weights=weights,
        )

        self.model.classifier[3] = nn.Linear(
            self.model.classifier[3].in_features,
            self.num_classes,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.preprocess = weights.transforms()

        print(
            f"Modèle chargé : {self.model_name}"
        )
        print(
            f"Nombre de classes : {self.num_classes}"
        )
        print(
            f"Taille d'entrée : {self.image_size}x{self.image_size}"
        )
        print("Mapping des classes :")

        for class_name, index in self.class_to_idx.items():
            print(f"  {index}: {class_name}")

        print("Modèle GreenCheck AI chargé avec succès.")

    def predict(self, image):
        """
        Effectue une prédiction sur une image.
        """

        image = self.preprocess(image)

        batch = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(batch)

        return output