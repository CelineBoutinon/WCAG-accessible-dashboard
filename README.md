![](logo.png)

# RÉALISER UN DASHBOARD INCLUSIF AVEC STREAMLIT

Projet realisé en juillet 2025 dans le cadre de ma formation Data Scientist avec CentraleSupélec/OpenClassrooms.

## Objectif du projet
"Prêt à dépenser" est une société financière qui propose des crédits à la consommation pour des personnes ayant peu ou pas d'historique de prêt. L’entreprise a mis en œuvre un outil de "scoring crédit" pour calculer la probabilité qu’un client fasse défaut sur son crédit, puis classifier la demande en crédit accordé ou refusé. Les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. "Prêt à dépenser" a donc décidé de développer un dashboard interactif pour que les chargés de relation client puissent expliquer de façon la plus transparente possible les décisions d’octroi de crédit, lors de rendez-vous avec eux. Cette volonté de transparence va tout à fait dans le sens des valeurs que l’entreprise veut incarner. "Prêt à dépenser" étant un employeur inclusif, l'application devra être adaptée aux utilisateurs ayant des déficiences visuelles ou motrices.



## Liste des fichiers

Pour le code de l'API et les tests, se référer au repo https://github.com/CelineBoutinon/credit-scoring-api.  
Pour le modèle de credit scoring, se référer au repo https://github.com/CelineBoutinon/credit-scoring.  
Les données brutes sont disponibles sur https://www.kaggle.com/c/home-credit-default-risk/data ou en téléchargement direct sur https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip.  

* **fichiers :**
  - **app.pi :** code Python de l'API Flask accessible sur https://credit-scoring-api-0p1u.onrender.com
  - **streamlit_cloud_app_P8_v1.py :** code Python de la page d'accueil de l'application Streamlit disponible sur https://ocds-p8-dashboard.streamlit.app/
  - **slideshow.pdf :** diapositives de présentation du projet
  - **merged_data_dico.csv :** dictionnaire des données accessibles sur les graphiques de l'application

* **dossiers :**
  - **pages :** code Python des 4 pages de l'application Streamlit

## Langages & software

 * python 3.9.13

Voir requirements.txt pour la liste complète des librairies & packages.



## Compétences développées
 * Réaliser la présentation orale d’une démarche de modélisation à un client interne/externe
 * Réaliser un tableau de bord inclusir et accessible afin de présenter son travail de modélisation à un public non technique et/ou ayant des déficiences visuelles ou motrices
 


## MENTIONS LÉGALES

Cette étude a été produite par CelineBoutinon sur la base du jeu de données Home Credit Default Risk sur Kaggle (https://www.kaggle.com/c/home-credit-default-risk/data). Le jeu de données est fourni « tel quel » et est hébergé sur Kaggle à des fins de recherche et d’éducation ; son utilisation est soumise aux conditions générales de Kaggle ainsi qu’aux termes disponibles sur https://www.kaggle.com/competitions/home-credit-default-risk/rules#7-competition-data. Les utilisateurs des données sont responsables de l’utilisation qu’ils en font et les analyses présentées ici restent la responsabilité seule de l'auteure. Pour plus de détails, veuillez consulter les conditions d’utilisation sur https://www.kaggle.com/terms et https://www.kaggle.com/docs/datasets#licensing.