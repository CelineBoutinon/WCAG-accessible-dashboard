# credit-scoring

![](logo.png)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projet realisé en mars 2025 dans le cadre de ma formation Data Scientist avec OpenClassrooms.

## Contexte
"Prêt à dépenser" est une société financière qui propose des crédits à la consommation pour
des personnes ayant peu ou pas d'historique de prêt. L’entreprise souhaite mettre en œuvre
un outil de “scoring crédit” pour calculer la probabilité qu’un client fasse défaut sur son crédit, 
puis classifier la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme
de classification en s’appuyant sur des sources de données variées (données comportementales,
données provenant d'autres institutions financières, etc...). Il s'agira donc de :  

- Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un
 client de façon automatique;
- Analyser les features qui contribuent le plus au modèle, d’une manière générale (feature
 importance globale) et au niveau d’un client (feature importance locale), afin de permettre
 à un chargé d’études de mieux comprendre le score attribué par le modèle et l'action qui
 lui est suggérée ("decline application" ou "grant loan");
- Mettre en production le modèle de scoring de prédiction à l’aide d’une API dans le cloud et réaliser
au préalable une interface locale de test de cette API;
- Mettre en œuvre une approche MLOps end-to-end, du tracking des expérimentations
à l’analyse en production du data drift.

Les données brutes sont disponibles ici: https://www.kaggle.com/c/home-credit-default-risk/data

## Objectif du projet

- Définir et mettre en œuvre un pipeline d’entraînement des modèles  
- Définir la stratégie d’élaboration d’un modèle d’apprentissage supervisé  
- Évaluer les performances des modèles d’apprentissage supervisé  
- Mettre en œuvre un logiciel de version de code  
- Suivre la performance d’un modèle en production et en assurer la maintenance  
- Concevoir un déploiement continu d'un moteur d’inférence sur une plateforme Cloud  

## Organisation du projet

Pour l'EDA et la modélisation, se référer au repo https://github.com/CelineBoutinon/credit-scoring.git

```
├── LICENSE                      <- Open-source license if one is chosen
├── README.md                    <- The top-level README for developers using this project.
├── .github/workflows
│   ├── test-app.yml             <- GitHub Actions unit tests execution script.
│
├── app.py                       <- Flask API
│
├── streamlit_cloud_app_vf.py    <- Streamlit app
│
├── pytest.ini                   <- Unit tests init file
│
├── requirements.txt             <- environment and dependencies
│
├── test_api.py                  <- PyTest unit tests script
│

```

--------

## Requirements

flask==3.1.0  
joblib==1.4.2  
pandas==2.2.3  
gunicorn==23.0.0  
imbalanced-learn==0.13.0  
imblearn==0.0  
scikit-learn==1.6.1  
mlflow==2.21.0  
shap==0.47.0  
lightgbm==4.5.0  
requests==2.32.3  
matplotlib==3.10.1  
pytest==8.3.5  
streamlit==1.44.1  
streamlit-shap==1.0.2    
