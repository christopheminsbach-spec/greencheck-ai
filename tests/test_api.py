from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

TEST_IMAGE = (
    Path(__file__).parent
    / "8b4f6b7d-28d4-4ab9-8a7e-e215300bdbee.jpg"
)


def test_predict_returns_prediction_and_recommendations():
    with TEST_IMAGE.open("rb") as image:
        response = client.post(
            "/predict",
            files={
                "file": (
                    TEST_IMAGE.name,
                    image,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["prediction"] == "Bougainvillea"
    assert data["model"] == "MobileNetV3-Small"
    assert data["model_status"] == "greencheck_trained"

    diagnosis = data["diagnosis"]

    assert diagnosis["prediction"] == data["prediction"]
    assert diagnosis["confidence"] == data["confidence"]
    assert diagnosis["status"] == "greencheck_trained"
    assert diagnosis["recommendations"]
    assert len(diagnosis["recommendations"]) == 3
    assert all(
        isinstance(recommendation, str)
        for recommendation in diagnosis["recommendations"]
    )


def test_predict_rejects_invalid_image():
    response = client.post(
        "/predict",
        files={
            "file": (
                "invalid.txt",
                b"ceci n'est pas une image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Le fichier envoyé n'est pas une image valide."
    )
