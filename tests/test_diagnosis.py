from app.diagnosis import RECOMMENDATIONS, get_recommendations


EXPECTED_COUNTS = {
    "Bougainvillea": 3,
    "Crown of thorns": 3,
    "Hibiscus": 3,
    "Jungle geranium": 2,
    "Madagascar periwinkle": 3,
    "Marigold": 3,
    "Rose": 3,
}


def test_all_greencheck_classes_have_recommendations():
    assert set(RECOMMENDATIONS) == set(EXPECTED_COUNTS)


def test_recommendation_counts():
    for plant, expected_count in EXPECTED_COUNTS.items():
        assert len(get_recommendations(plant)) == expected_count


def test_recommendations_are_strings():
    for plant in EXPECTED_COUNTS:
        recommendations = get_recommendations(plant)

        assert recommendations
        assert all(
            isinstance(recommendation, str)
            for recommendation in recommendations
        )


def test_unknown_class_returns_empty_list():
    assert get_recommendations("Unknown plant") == []
