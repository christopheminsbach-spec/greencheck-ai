from pydantic import BaseModel


class DiagnosisResult(BaseModel):
    """
    Structure du résultat métier d'un diagnostic GreenCheck.
    """

    status: str
    prediction: str
    confidence: float
    diagnosis: str
    recommendations: list[str]
