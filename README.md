# GreenCheck AI 🌱

Microservice d'intelligence artificielle du projet **GreenCheck**, développé dans le cadre du projet CD2IA 2025/2026 pour AgroNovaTech.

GreenCheck a pour objectif de proposer un diagnostic intelligent de plantes à partir d'une photographie.

## 🎯 Objectif

Le service `greencheck-ai` fournit une API REST permettant à terme de :

1. recevoir une image de plante ;
2. analyser l'image avec un modèle d'intelligence artificielle ;
3. identifier la plante ou une éventuelle maladie ;
4. retourner une classe prédite ;
5. fournir un niveau de confiance ;
6. associer des recommandations adaptées au résultat.

Le microservice est indépendant de l'application Odoo GreenCheck.

## 🏗️ Architecture

```text
Utilisateur
    │
    ▼
GreenCheck Odoo 19
    │
    │ API
    ▼
GreenCheck AI
FastAPI :8001
    │
    ▼
MobileNetV3-Small
    │
    ▼
Classification d'image
    │
    ├── Classe prédite
    ├── Confiance
    └── Recommandations
```

## 🛠️ Technologies

* Python 3.12
* FastAPI
* Uvicorn
* PyTorch
* Torchvision
* MobileNetV3-Small
* Docker

## 📁 Structure actuelle

```text
greencheck-ai/
├── app/
│   └── main.py
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🚀 Lancer le projet avec Docker

Construire l'image :

```bash
docker build -t greencheck-ai .
```

Lancer le service :

```bash
docker run --rm -p 8001:8001 --name greencheck-ai greencheck-ai
```

L'API est alors accessible à :

```text
http://localhost:8001
```

Documentation Swagger :

```text
http://localhost:8001/docs
```

Documentation ReDoc :

```text
http://localhost:8001/redoc
```

## 🧪 État actuel du projet

### FastAPI

Le service FastAPI est fonctionnel.

L'endpoint `/` retourne actuellement :

```json
{
  "service": "GreenCheck AI",
  "status": "ok",
  "message": "FastAPI fonctionne"
}
```

### PyTorch

PyTorch est installé et fonctionnel dans l'image Docker.

Le fonctionnement CPU est actuellement utilisé pour l'inférence.

### MobileNetV3-Small

Le modèle pré-entraîné `MobileNetV3-Small` de Torchvision a été chargé avec succès dans le conteneur Docker.

Le modèle contient environ **2,54 millions de paramètres**.

L'intégration du modèle dans l'API FastAPI reste à réaliser.

## 🔬 Prochaines étapes

Le développement sera réalisé progressivement :

```text
1. API FastAPI de base                 ✅
2. Docker                             ✅
3. PyTorch                            ✅
4. Torchvision                        ✅
5. MobileNetV3-Small                  ✅ test de chargement
6. Endpoint d'analyse d'image          ⏳
7. Prétraitement des images            ⏳
8. Inférence MobileNetV3-Small         ⏳
9. Retour classe + confiance           ⏳
10. Règles de recommandations          ⏳
11. Tests automatisés                  ⏳
12. Intégration avec Odoo              ⏳
```

## 🌱 Données et entraînement

Dans un premier temps, le modèle est utilisé avec ses poids pré-entraînés.

L'entraînement ou le fine-tuning sur un dataset spécialisé dans les plantes sera réalisé dans une étape ultérieure.

Le projet pourra notamment utiliser des jeux de données spécialisés tels que PlantVillage pour l'entraînement initial et des données plus réalistes pour l'évaluation.

## 🔐 Sécurité

Les futures versions devront notamment prévoir :

* validation du type de fichier ;
* limitation de la taille des images ;
* validation des entrées API ;
* gestion des erreurs ;
* limitation des requêtes ;
* protection des données utilisateur ;
* prise en compte des exigences RGPD.

## 📚 Documentation API

FastAPI génère automatiquement la documentation OpenAPI.

Une fois le serveur lancé :

* Swagger UI : `/docs`
* ReDoc : `/redoc`
* OpenAPI : `/openapi.json`

## 📌 Projet associé

Ce microservice constitue le moteur IA du projet **GreenCheck**.

```text
greencheck-odoo/
        │
        │ API
        ▼
greencheck-ai/
        │
        ▼
MobileNetV3-Small

