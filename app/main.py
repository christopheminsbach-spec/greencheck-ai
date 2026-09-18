from fastapi import FastAPI


app = FastAPI(
    title="GreenCheck AI",
    description="API d'intelligence artificielle pour le diagnostic des plantes",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "service": "GreenCheck AI",
        "status": "ok",
        "message": "FastAPI fonctionne"
    }