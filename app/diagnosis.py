from pydantic import BaseModel


RECOMMENDATIONS = {
    "Bougainvillea": [
        "Exposition très lumineuse et ensoleillée.",
        "Maintenir un sol bien drainé.",
        "Arroser modérément et laisser sécher partiellement le substrat entre deux arrosages.",
    ],
    "Crown of thorns": [
        "Exposition lumineuse avec plusieurs heures de soleil.",
        "Utiliser un substrat bien drainant.",
        "Arroser modérément et éviter l'excès d'eau.",
    ],
    "Hibiscus": [
        "Forte luminosité, idéalement avec du soleil.",
        "Maintenir un arrosage régulier sans détremper le sol.",
        "Surveiller le dessèchement du substrat.",
    ],
    "Jungle geranium": [
        "Installer dans un environnement lumineux et chaud.",
        "Maintenir une humidité régulière du sol tout en assurant un bon drainage.",
    ],
    "Madagascar periwinkle": [
        "Préférer une exposition lumineuse.",
        "Utiliser un sol drainant.",
        "Arroser régulièrement mais éviter l'eau stagnante.",
    ],
    "Marigold": [
        "Privilégier une exposition ensoleillée.",
        "Utiliser un sol correctement drainé.",
        "Arroser lorsque le substrat commence à sécher.",
    ],
    "Rose": [
        "Exposition ensoleillée.",
        "Assurer un sol fertile et bien drainé.",
        "Arroser régulièrement au pied et éviter de maintenir le feuillage constamment humide.",
    ],
}


def get_recommendations(prediction: str) -> list[str]:
    """
    Retourne les recommandations associées à l'espèce identifiée.
    """

    return RECOMMENDATIONS.get(prediction, [])


class DiagnosisResult(BaseModel):
    """
    Structure du résultat métier d'un diagnostic GreenCheck.
    """

    status: str
    prediction: str
    confidence: float
    diagnosis: str
    recommendations: list[str]
