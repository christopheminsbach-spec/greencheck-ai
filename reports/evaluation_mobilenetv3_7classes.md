# GreenCheck AI — Évaluation finale MobileNetV3-Small

## 1. Objet

Ce document fige les résultats de l'évaluation finale du modèle de classification GreenCheck AI sur le jeu de données de test indépendant.

L'évaluation est réalisée après l'entraînement et utilise exclusivement le jeu `test`.

Aucun réentraînement n'est effectué pendant cette phase.

## 2. Modèle évalué

- Architecture : MobileNetV3-Small
- Nombre de classes : 7
- Taille d'entrée : 224 × 224 pixels
- Checkpoint : `models/mobilenetv3_small_7classes_best.pth`
- Epoch sélectionné : 7
- Phase : 2
- Validation loss du checkpoint : `0.021704915268350708`
- Validation accuracy du checkpoint : `99.38271604938271 %`
- Device d'évaluation : CPU

## 3. Classes

Le mapping utilisé est :

| Index | Classe |
|---:|---|
| 0 | Bougainvillea |
| 1 | Crown of thorns |
| 2 | Hibiscus |
| 3 | Jungle geranium |
| 4 | Madagascar periwinkle |
| 5 | Marigold |
| 6 | Rose |

Le mapping du checkpoint et celui du dataset `test` ont été vérifiés avant l'évaluation.

## 4. Dataset de test

- Dataset : `datasets/tropical_flower/split_clean/test`
- Nombre total d'images : **647**
- Nombre de classes : **7**
- Jeu utilisé exclusivement pour l'évaluation finale.

Transformation appliquée :

- Resize 256
- CenterCrop 224
- ToTensor
- Normalisation ImageNet

## 5. Résultats globaux

| Métrique | Résultat |
|---|---:|
| Images testées | 647 |
| Prédictions correctes | 643 |
| Prédictions incorrectes | 4 |
| Test loss | 0.019578 |
| Test accuracy | **99.3818 %** |

## 6. Résultats par classe

| Classe | Correctes | Incorrectes | Accuracy |
|---|---:|---:|---:|
| Bougainvillea | 87/87 | 0 | 100.0000 % |
| Crown of thorns | 86/88 | 2 | 97.7273 % |
| Hibiscus | 82/82 | 0 | 100.0000 % |
| Jungle geranium | 102/104 | 2 | 98.0769 % |
| Madagascar periwinkle | 55/55 | 0 | 100.0000 % |
| Marigold | 107/107 | 0 | 100.0000 % |
| Rose | 124/124 | 0 | 100.0000 % |

## 7. Matrice de confusion

Lignes = classe réelle
Colonnes = classe prédite

| Réel \ Prédit | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 — Bougainvillea | 87 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 — Crown of thorns | 1 | 86 | 0 | 1 | 0 | 0 | 0 |
| 2 — Hibiscus | 0 | 0 | 82 | 0 | 0 | 0 | 0 |
| 3 — Jungle geranium | 0 | 0 | 1 | 102 | 0 | 1 | 0 |
| 4 — Madagascar periwinkle | 0 | 0 | 0 | 0 | 55 | 0 | 0 |
| 5 — Marigold | 0 | 0 | 0 | 0 | 0 | 107 | 0 |
| 6 — Rose | 0 | 0 | 0 | 0 | 0 | 0 | 124 |

## 8. Erreurs observées

Les 4 erreurs sont :

1. `Crown of thorns` → `Bougainvillea`
2. `Crown of thorns` → `Jungle geranium`
3. `Jungle geranium` → `Hibiscus`
4. `Jungle geranium` → `Marigold`

## 9. Conclusion factuelle

L'évaluation finale sur le jeu de test indépendant comporte 647 images.

Le modèle obtient :

**99.3818 % d'accuracy**, soit **643 prédictions correctes sur 647**.

Six classes sur sept ne présentent aucune erreur sur ce jeu de test.

Les erreurs observées concernent uniquement les classes `Crown of thorns` et `Jungle geranium`.

Ces résultats constituent la référence d'évaluation du modèle `MobileNetV3-Small` à 7 classes pour GreenCheck AI à la date du 25 septembre 2026.
