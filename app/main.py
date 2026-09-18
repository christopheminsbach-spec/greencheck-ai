from fastapi import FastAPI

from app.model import PlantModel


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