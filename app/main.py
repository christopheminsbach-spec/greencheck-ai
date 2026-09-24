from io import BytesIO

import torch
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image

from app.model import PlantModel


class PredictResponse(BaseModel):
    """
    Structure de réponse de l'endpoint de prédiction GreenCheck AI.
    """

    success: bool
    filename: str
    prediction: str
    confidence: float
    model: str
    model_status: str


app = FastAPI(
    title="GreenCheck AI",
    description="API d'intelligence artificielle pour le diagnostic des plantes",
    version="0.1.0",
)


# Chargement du modèle une seule fois au démarrage de l'API
plant_model = PlantModel()


@app.get("/")
def root():
    return {
        "service": "GreenCheck AI",
        "status": "ok",
        "message": "FastAPI fonctionne"
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Analyse une image avec MobileNetV3-Small
    et retourne la classe prédite avec sa confiance.
    """

    image_data = await file.read()

    image = Image.open(
        BytesIO(image_data)
    ).convert("RGB")

    output = plant_model.predict(image)

    probabilities = torch.nn.functional.softmax(
        output,
        dim=1
    )

    confidence, class_id = torch.max(
        probabilities,
        dim=1
    )

    categories = plant_model.weights.meta["categories"]

    predicted_class = categories[class_id.item()]

    return {
        "success": True,
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": round(confidence.item(), 4),
        "model": "MobileNetV3-Small",
        "model_status": "pretrained_imagenet",
    }
