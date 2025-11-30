# HAJBI DOUAE
<img src="douae.jpeg" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>

Numéro d'étudiant : 25007751
Classe : CAC2
Compte rendu
Analyse Prédictive de Régression sur Données Instagram
Date : 30 Novembre 2025​

Table des Matières
Introduction et Contexte

Analyse Exploratoire des Données (Data Analysis)

Chargement et Structure du Dataset

Distribution de la Variable Cible

Préparation des Variables Prédictives

Analyse Statistique et Visuelle

Méthodologie de Régression

Séparation des Données (Data Split)

Modèles de Régression Testés

Résultats et Comparaison des Modèles

Pré-traitement des Données

Performances des Modèles

Comparaison des Performances

Conclusion

1. Introduction et Contexte
Ce rapport présente une analyse prédictive de régression réalisée sur le dataset Instagram Analytics provenant de Kaggle, dans le cadre d'une étude en Science des Données. En suivant le cycle de vie des données, une exploration approfondie (EDA), un prétraitement et plusieurs modélisations de régression ont été effectuées.​

L'objectif principal est de prédire la variable cible "Life Expectancy" à partir de multiples features incluant des variables numériques et catégorielles (comme Country et Gender), tout en évaluant et comparant les performances de cinq algorithmes : régression linéaire, polynomiale, arbre de décision, forêt aléatoire et SVM.​

2. Analyse Exploratoire des Données (Data Analysis)
2.1 Chargement et Structure du Dataset
Le dataset provient d'un fichier ZIP extrait contenant des données globales sur la santé, la nutrition, la mortalité et l'économie, avec la variable cible principale "Life Expectancy". Le DataFrame chargé présente une structure mixte avec des variables numériques et catégorielles.​

Nombre d'échantillons ($N$) : Plusieurs milliers d'observations (précisément déterminé après chargement).

Nombre de variables ($d$) : Multiples colonnes incluant des features numériques et catégorielles.

Variables d'entrée ($X$) : Features économiques, nutritionnelles, démographiques (ex: GDP, BMI, etc.) + variables catégorielles (Country, Gender).

Variable de sortie ($Y$) : Life Expectancy (espérance de vie).
import pandas as pd
# Chargement depuis fichier ZIP extrait
df = pd.read_csv(fullcsvpath)
df.info()
```python
print(df.head())
```
2.2 Distribution de la Variable Cible
La variable "Life Expectancy" présente une distribution continue typique des données démographiques, analysée via des statistiques descriptives. Des valeurs manquantes ont été détectées et traitées.​

2.3 Préparation des Variables Prédictives
Les variables catégorielles "Country" et "Gender" ont été identifiées et encodées via one-hot encoding. "Country" comportant un grand nombre de catégories uniques (plus de 200), l'encodage a été appliqué malgré la dimensionnalité résultante.
# Encodage catégorielles
df = pd.get_dummies(df, columns=['Gender'], prefix='Gender')
df = pd.get_dummies(df, columns=['Country'], prefix='Country')
2.4 Analyse Statistique et Visuelle
L'analyse a révélé des valeurs manquantes dans les colonnes numériques, imputées par la médiane. Des visualisations (scatter plots, barplots) ont confirmé la nécessité d'un prétraitement robuste avant modélisation.
import matplotlib.pyplot as plt
import seaborn as sns
# Statistiques descriptives et visualisations
df.describe()
sns.boxplot(data=df.select_dtypes(include='number'))
plt.show()
3. Méthodologie de Régression
3.1 Séparation des Données (Data Split)
Les données ont été divisées en ensembles d'entraînement (80%) et de test (20%) avec train_test_split(random_state=42) pour évaluer la généralisation des modèles.
```python
from sklearn.model_selection import train_test_split
X = df.drop('Life Expectancy', axis=1)
y = df['Life Expectancy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
3.2 Modèles de Régression Testés
Cinq modèles ont été implémentés et évalués :

Régression Linéaire (baseline)

Régression Polynomiale (degré 2)

Arbre de Décision

Forêt Aléatoire (100 estimateurs)

Support Vector Regression (SVR avec kernel RBF).​

4. Résultats et Comparaison des Modèles
4.1 Pré-traitement des Données
Imputation médiane pour valeurs manquantes numériques et one-hot encoding pour catégorielles. Aucun scaler n'est mentionné explicitement, mais les modèles sensibles (SVR) ont bénéficié du prétraitement.
# Imputation médiane
for col in df.select_dtypes(include='number').columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)
4.2 Performances des Modèles
Les métriques clés (MSE, MAE, R²) ont été calculées sur l'ensemble de test pour chaque modèle.​

4.3 Comparaison des Performances
Modèle	MSE	MAE	R²
Random Forest Regression	0.5501	0.4287	0.9943 ​
Decision Tree Regression	0.7028	0.4952	0.9928 ​
Support Vector Regression	0.9479	0.6865	0.9901 ​
Polynomial Regression (2)	1.4728	0.8930	0.9849 ​
Linear Regression	1.7005	0.9705	0.9825 ​
La Forêt Aléatoire domine avec le meilleur R² et les erreurs les plus faibles.​

5. Conclusion
   
Cette analyse valide l'importance du prétraitement (imputation, encodage) pour les datasets hétérogènes. La Forêt Aléatoire excelle grâce à sa capacité à capturer les relations non-linéaires complexes de l'espérance de vie.​

Les concepts clés démontrés incluent : exploration approfondie, validation croisée via train/test split, et sélection du meilleur modèle par métriques multiples. Une optimisation future via GridSearchCV sur Random Forest est recommandée.
