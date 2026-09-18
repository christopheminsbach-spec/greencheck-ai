import torch
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


class PlantModel:
    """
    Gestion du modèle MobileNetV3-Small utilisé par GreenCheck AI.
    """

    def __init__(self):
        print("Chargement de MobileNetV3-Small...")

        self.weights = MobileNet_V3_Small_Weights.DEFAULT

        self.model = mobilenet_v3_small(
            weights=self.weights
        )

        self.model.eval()

        self.preprocess = self.weights.transforms()

        print("MobileNetV3-Small chargé avec succès.")

    def predict(self, image):
        """
        Effectue une prédiction sur une image.

        Cette méthode sera complétée lors de l'étape
        d'intégration de l'endpoint d'analyse.
        """

        image = self.preprocess(image)

        batch = image.unsqueeze(0)

        with torch.no_grad():
            output = self.model(batch)

        return output