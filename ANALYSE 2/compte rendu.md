# HAJBI DOUAE
<img src="douae.jpeg" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>

Numéro d'étudiant : 25007751
Classe : CAC2
Compte rendu
Analyse Prédictive des Prix de Vente des Voitures par Régression
Date : 03 Décembre 2025
Table des Matières

Introduction et Contexte
Analyse Exploratoire des Données (Data Analysis)
Chargement et Structure du Dataset
Statistiques Descriptives et Valeurs Manquantes
Création de la Variable car_age et Encodage
Nettoyage des Doublons et Séparation des Données

Méthodologie de Régression
Modèles Linéaires (Ridge, Lasso, Linéaire)
Modèles Basés sur les Arbres (Arbre de Décision, Forêt Aléatoire)
Modèles Non Linéaires (Polynomiale, Gradient Boosting)

Résultats et Comparaison des Modèles
Performances Individuelles
Visualisations des Prédictions
Comparaison Globale

Conclusion


1. Introduction et Contexte
Ce rapport présente une analyse détaillée d'un jeu de données réel concernant les détails des voitures d'occasion, réalisée dans le cadre d'une étude en Science des Données. En suivant le cycle de vie des données, nous avons mené une exploration (EDA), un prétraitement et une modélisation prédictive.
L'objectif est de construire des modèles de régression capables de prédire le prix de vente (selling_price) des voitures en utilisant divers algorithmes, et d'évaluer leurs performances en termes de MAE, MSE et R² pour identifier le plus efficace.
2. Analyse Exploratoire des Données (Data Analysis)
2.1 Chargement et Structure du Dataset
Le jeu de données CAR DETAILS FROM CAR DEKHO.csv contient les caractéristiques des voitures d'occasion vendues en Inde.

Nombre d'échantillons ($  N  $) : 3577 observations (avant nettoyage).
Nombre de variables ($  d  $) : 8 colonnes initiales.
Variables d'entrée ($  X  $) :name, year, km_driven, fuel, seller_type, transmission, owner.
Variable de sortie ($  Y  $) :selling_price (en roupies indiennes). 
```python
import pandas as pd
df = pd.read_csv('/content/CAR DETAILS FROM CAR DEKHO.csv')
print("========= Résumé du Dataset =========")
df.info()
print("\n========= Premiers échantillons =========")
print(df.head())
```
2.2 Statistiques Descriptives et Valeurs Manquantes
L'analyse des statistiques descriptives montre une large gamme de prix et de kilomètres parcourus. Aucune valeur manquante n'a été détectée.

Statistiques,selling_price,km_driven,...
count,3577,3577,...
mean,504127.311745,66215.777933,...
std,578548.736139,46623.385991,...
min,20000,350,...
25%,220000,36000,...
50%,350000,60000,...
75%,600000,90000,...
max,8900000,806599,...

Nombre de doublons : 79.
Nombre de noms de voitures uniques : 1491.
Top 10 noms uniques : ['Maruti 800 AC', 'Maruti Wagon R LXI Minor', ...].

```python
print("\nDescriptive Statistics:")
print(df.describe())
print("\nMissing Values per column:")
print(df.isnull().sum())
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())
print("\nNumber of unique car names:", df['name'].nunique())
print("Top 10 unique car names:", df['name'].unique()[:10])
```
2.3 Création de la Variable car_age et Encodage
Pour mieux capturer l'âge des voitures, nous avons créé car_age = année_courante - year (année courante : 2024).
Suppression de year.
Encodage one-hot des variables catégorielles : fuel, seller_type, transmission, owner.
```python
import datetime
# Create 'car_age' feature
current_year = datetime.datetime.now().year
df['car_age'] = current_year - df['year']
# Drop the original 'year' column as 'car_age' replaces it
df.drop('year', axis=1, inplace=True)
# Identify categorical columns for one-hot encoding (excluding 'name' for now)
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
# Apply one-hot encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("DataFrame after creating 'car_age' and one-hot encoding categorical variables:")
print(df.head())
print("\nDataFrame Info after preprocessing:")
df.info()
```
2.4 Nettoyage des Doublons et Séparation des Données
Suppression des 79 doublons → 3498 échantillons uniques.
Séparation en ensembles d'entraînement (80%) et de test (20%).
Forme de X_train : (2798, 11), X_test : (700, 11).
print(f"Number of rows before dropping duplicates: {df.shape[0]}")
df.drop_duplicates(inplace=True)
```python
print(f"Number of rows after dropping duplicates: {df.shape[0]}")
# Separate target variable (y) and features (X)
X = df.drop(['selling_price'], axis=1)  # 'name' column was already dropped in a previous step
y = df['selling_price']
# Import train_test_split
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")
print("First 5 rows of X_train:")
print(X_train.head())
```
(Les étapes de nettoyage et de prétraitement ont assuré une qualité des données optimale pour la modélisation, en évitant les biais dus aux doublons et en convertissant les catégories en numériques.)
3. Méthodologie de Régression
3.1 Modèles Linéaires (Ridge, Lasso, Linéaire)
Nous avons entraîné des modèles linéaires avec régularisation (alpha=1.0) pour minimiser les erreurs et éviter le surapprentissage.
3.2 Modèles Basés sur les Arbres (Arbre de Décision, Forêt Aléatoire)
Ces modèles non paramétriques capturent les interactions complexes sans hypothèse de linéarité (random_state=42 pour reproductibilité).
3.3 Modèles Non Linéaires (Polynomiale, Gradient Boosting)

Polynomiale (degré 2) : Transformation des features pour capturer les non-linéarités.
Gradient Boosting : Ensemble séquentiel pour corriger les erreurs itérativement (random_state=42).
Tous les modèles ont été entraînés sur X_train/y_train et évalués sur X_test/y_test.


4. Résultats et Comparaison des Modèles
4.1 Performances Individuelles
Sur les données préparées, les modèles ont montré des performances variées, avec des évaluations via MAE, MSE et R².
Exemple pour Ridge :
```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Initialize dictionaries to store predictions and metrics if they don't exist
if 'predictions' not in locals():
    predictions = {}
if 'metrics' not in locals():
    metrics = {}
# Instantiate and train the Ridge model
ridge_model = Ridge(alpha=1.0)  # You can adjust alpha as needed
ridge_model.fit(X_train, y_train)
# Make predictions on the test set
y_pred_ridge = ridge_model.predict(X_test)
# Calculate evaluation metrics
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"--- Ridge Regression Model Performance ---")
print(f"Mean Absolute Error (MAE): {mae_ridge:,.2f}")
print(f"Mean Squared Error (MSE): {mse_ridge:,.2f}")
print(f"R-squared (R2): {r2_ridge:.4f}")
# Store predictions for later use if needed
predictions["Ridge Regression"] = y_pred_ridge
metrics["Ridge Regression"] = {"MAE": mae_ridge, "MSE": mse_ridge, "R2": r2_ridge}
```
MAE: 212,525.95 ; MSE: 194,565,631,338.84 ; R²: 0.3962.
Performances similaires pour Lasso et Linéaire.
Arbre de Décision : R² faible (0.0450).
Forêt Aléatoire : R² 0.3498.
Polynomiale : R² 0.4755.
Gradient Boosting : Meilleur R² (0.4799).
4.2 Visualisations des Prédictions
Les scatter plots montrent la dispersion des prédictions vs réelles, avec une ligne diagonale pour les prédictions parfaites.
Exemple pour Ridge :
```python
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_ridge, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Prédictions parfaites')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions Ridge')
plt.title('Prédictions du Modèle Ridge vs. Valeurs Réelles')
plt.legend()
plt.grid(True)
```
Les points sont dispersés, avec plus de variabilité pour les prix élevés. Gradient Boosting montre la meilleure concentration autour de la diagonale.
Méthode,MAE,MSE,R²
Gradient Boosting,183 842.82,167 586 152 554.34,0.4799
Polynomiale,188 233.43,169 018 006 400.79,0.4755
Linéaire,212 564.76,194 539 409 801.70,0.3963
Lasso,212 564.92,194 539 688 463.72,0.3963
Ridge,212 525.95,194 565 631 338.84,0.3962
Forêt Aléatoire,207 922.77,209 513 198 128.50,0.3498
Arbre de Décision,247 321.00,307 727 907 979.00,0.0450
```python
import pandas as pd
# Create a DataFrame from the metrics dictionary
metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
# Sort the DataFrame by R2 score in descending order
metrics_df_sorted = metrics_df.sort_values(by='R2', ascending=False)
print("--- Comparaison des performances des modèles de régression ---")
print(metrics_df_sorted.to_markdown(numalign="left", stralign="left"))
```
Gradient Boosting surpasse les autres, indiquant des relations non linéaires. Les modèles linéaires sont similaires, sans gain notable de régularisation.
5. Conclusion
Ce projet a permis de valider plusieurs concepts clés en Data Science :
1. Exploration : Comprendre la structure et nettoyer les données (doublons, encodage) est crucial avant la modélisation.
2. Prétraitement : La création de features comme car_age et l'encodage one-hot sont indispensables pour les algorithmes de régression.
3. Méthodologie : La comparaison de multiples modèles (linéaires à non linéaires) et l'évaluation rigoureuse (MAE, MSE, R²) permettent d'identifier le meilleur (Gradient Boosting) et d'éviter les biais.
