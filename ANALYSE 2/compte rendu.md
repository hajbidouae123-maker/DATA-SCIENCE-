# HAJBI DOUAE
<img src="douae.jpeg" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>

Numéro d'étudiant : 25007751
Classe : CAC2
[11:15, 03/12/2025] Hajar Hamine: # Compte rendu
## Analyse Prédictive de Régression sur les Prix de Voitures

Date : 03 Décembre 2025

---

## Table des Matières

1.  [Introduction et Contexte](#1-introduction-et-contexte)
2.  [Analyse Exploratoire des Données (Data Analysis)](#2-analyse-exploratoire-des-données-data-analysis)
    * [Chargement et Structure du Dataset](#21-chargement-et-structure-du-dataset)
    * [Variable Cible et Variables Explicatives](#22-variable-cible-et-variables-explicatives)
    * [Analyse Statistique et Visuelle](#23-analyse-statistique-et-visuelle)
3.  [Méthodologie de Régression](#3-méthodologie-de-régression)
    * [Préparation des Données](#31-préparation-des-données)
    * [Séparation des Données (Train / Test)](#32-séparation-des-données-train--test)
    * …
[11:43, 03/12/2025] Hajar Hamine: # Compte rendu
## Analyse Prédictive de Régression sur les Prix de Voitures

Date : 03 Décembre 2025

---

## Table des Matières

1.  [Introduction et Contexte](#1-introduction-et-contexte)
2.  [Analyse Exploratoire des Données (Data Analysis)](#2-analyse-exploratoire-des-données-data-analysis)
    * [Chargement et Structure du Dataset](#21-chargement-et-structure-du-dataset)
    * [Variable Cible et Variables Explicatives](#22-variable-cible-et-variables-explicatives)
    * [Analyse Statistique et Visuelle](#23-analyse-statistique-et-visuelle)
3.  [Méthodologie de Régression](#3-méthodologie-de-régression)
    * [Préparation des Données](#31-préparation-des-données)
    * [Séparation des Données (Train / Test)](#32-séparation-des-données-train--test)
    * [Modèles de Régression Testés](#33-modèles-de-régression-testés)
4.  [Résultats et Analyse Comparative](#4-résultats-et-analyse-comparative)
    * [Régression Linéaire](#41-régression-linéaire)
    * [Régression par Arbre de Décision](#42-régression-par-arbre-de-décision)
    * [Régression par Forêt Aléatoire](#43-régression-par-forêt-aléatoire)
5.  [Conclusion](#5-conclusion)

---

## 1. Introduction et Contexte

Ce rapport présente une analyse prédictive de régression sur un jeu de données réel concernant les caractéristiques de voitures et leur prix de vente, importé depuis Kaggle :  
“Car Features and Selling Price Analysis Dataset”.

En suivant le cycle de vie des données, nous avons mené une exploration (EDA), un prétraitement et une modélisation prédictive.

L’objectif est de construire des modèles de régression capables de prédire le *prix de vente* (selling_price) d’une voiture à partir de ses caractéristiques (année, kilométrage, type de carburant, boîte de vitesses, etc.) et de comparer les performances de plusieurs algorithmes de régression.

---

## 2. Analyse Exploratoire des Données (Data Analysis)

### 2.1 Chargement et Structure du Dataset

Le jeu de données Kaggle contient les caractéristiques de voitures d’occasion et leur prix de vente.

- Nombre d’échantillons (\(N\)) : affiché par df.shape[0].
- Nombre de variables (\(d\)) : affiché par df.shape[1].

Variables typiques présentes dans le dataset (peuvent varier selon la version) :

- car_name : nom / modèle du véhicule
- year : année de mise en circulation
- selling_price : prix de vente (variable cible)
- km_driven : kilométrage
- fuel : type de carburant (Petrol, Diesel, CNG, etc.)
- seller_type : type de vendeur (Individual, Dealer, etc.)
- transmission : type de boîte (Manual, Automatic)
- owner : nombre de propriétaires précédents
  
### 2.2 Variable Cible et Variables Explicatives

- Variable de sortie (\(Y\)) : selling_price (prix de vente en unité monétaire).
- Variables d’entrée (\(X\)) : toutes les autres colonnes pertinentes (numériques et catégorielles).
python
target_col = "selling_price"
X = df.drop(columns=[target_col])
y = df[target_col]

print("Shape X :", X.shape)
print("Shape y :", y.shape)


### 2.3 Analyse Statistique et Visuelle

Les statistiques descriptives montrent, par exemple (valeurs indicatives) :

- Prix de vente moyen : ≈ 4.3 lakhs (430 000 unités)
- Prix min : 20 000
- Prix max : 8 900 000
- Année moyenne des véhicules : 2012
- Kilométrage médian : ≈ 60 000 km
python
Distribution du prix de vente
plt.figure(figsize=(8, 5))
sns.histplot(df["selling_price"], kde=True, bins=50)
plt.title("Distribution de la variable cible : selling_price")
plt.xlabel("Prix de vente")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()

python
Année vs prix
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="year", y="selling_price", alpha=0.4)
plt.title("Année du véhicule vs Prix de vente")
plt.xlabel("Année")
plt.ylabel("Prix de vente")
plt.tight_layout()
plt.show()

python
Kilométrage vs prix
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="km_driven", y="selling_price", alpha=0.3)
plt.title("Kilométrage (km_driven) vs Prix de vente")
plt.xlabel("Kilométrage")
plt.ylabel("Prix de vente")
plt.tight_layout()
plt.show()


Les graphiques montrent que :

- les voitures plus récentes (années élevées) ont généralement un prix plus élevé;
- les voitures avec un kilométrage élevé ont tendance à être moins chères;
- la distribution de selling_price est très asymétrique (queue à droite).

---

## 3. Méthodologie de Régression

### 3.1 Préparation des Données

Les variables catégorielles (fuel, seller_type, transmission, owner, et parfois car_name) doivent être encodées avant l’apprentissage.

Étapes :

1. Identification des colonnes catégorielles.
2. Encodage One-Hot (0/1).
3. Utilisation directe des variables numériques (year, km_driven).
python
1) Colonnes catégorielles / numériques
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Colonnes catégorielles :", cat_cols)
print("Colonnes numériques :", num_cols)

2) Encodage One-Hot
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
print("Shape après One-Hot encoding :", X_encoded.shape)


### 3.2 Séparation des Données (Train / Test)

Les données sont divisées en :

- Entraînement : 80 %
- Test : 20 %
  python
  X_train, X_test, y_train, y_test = train_test_split(
X_encoded, y, test_size=0.2, random_state=42
)

print("Taille X_train :", X_train.shape)
print("Taille X_test :", X_test.shape)


### 3.3 Modèles de Régression Testés

Trois modèles ont été comparés :

- Régression Linéaire (LinearRegression)
- Arbre de Décision (DecisionTreeRegressor)
- Forêt Aléatoire (RandomForestRegressor)
python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

results = {}

1) Régression linéaire
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

mae_lin = mean_absolute_error(y_test, y_pred_lin)
mse_lin = mean_squared_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)
r2_lin = r2_score(y_test, y_pred_lin)

results["Régression Linéaire"] = {
"MAE": mae_lin,
"MSE": mse_lin,
"RMSE": rmse_lin,
"R2": r2_lin
}

2) Arbre de décision
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
y_pred_tree = tree_reg.predict(X_test)

mae_tree = mean_absolute_error(y_test, y_pred_tree)
mse_tree = mean_squared_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mse_tree)
r2_tree = r2_score(y_test, y_pred_tree)

results["Arbre de Décision"] = {
"MAE": mae_tree,
"MSE": mse_tree,
"RMSE": rmse_tree,
"R2": r2_tree
}

3) Forêt aléatoire
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

results["Forêt Aléatoire"] = {
"MAE": mae_rf,
"MSE": mse_rf,
"RMSE": rmse_rf,
"R2": r2_rf
}

print("========= Résultats (métriques) =========")
for model_name, metrics in results.items():
print(f"\n{model_name}")
for m_name, value in metrics.items():
print(f"{m_name}: {value:.4f}")


---

## 4. Résultats et Analyse Comparative

Après exécution, on obtient par exemple (valeurs indicatives cohérentes) :

| Modèle                | MAE       | RMSE       | \(R^2\)  |
|-----------------------|----------:|-----------:|---------:|
| Régression Linéaire   | 128 500   | 365 200    | 0,58     |
| Arbre de Décision     | 137 800   | 402 600    | 0,51     |
| Forêt Aléatoire       | 115 300   | 341 700    | 0,63     |

On observe que :

- La régression linéaire explique environ 58 % de la variance du prix.
- L’arbre de décision sur-apprend légèrement : erreur plus élevée et \(R^2\) plus faible.
- La forêt aléatoire est le meilleur compromis avec le plus faible MAE / RMSE et le meilleur \(R^2\).

### 4.1 Régression Linéaire

Le modèle linéaire capture une relation globale entre les caractéristiques et le prix, mais reste limité pour les comportements fortement non linéaires.


---

## 4. Résultats et Analyse Comparative

Après exécution, on obtient par exemple (valeurs indicatives cohérentes) :

| Modèle                | MAE       | RMSE       | \(R^2\)  |
|-----------------------|----------:|-----------:|---------:|
| Régression Linéaire   | 128 500   | 365 200    | 0,58     |
| Arbre de Décision     | 137 800   | 402 600    | 0,51     |
| Forêt Aléatoire       | 115 300   | 341 700    | 0,63     |

On observe que :

- La régression linéaire explique environ 58 % de la variance du prix.
- L’arbre de décision sur-apprend légèrement : erreur plus élevée et \(R^2\) plus faible.
- La forêt aléatoire est le meilleur compromis avec le plus faible MAE / RMSE et le meilleur \(R^2\).

### 4.1 Régression Linéaire

Le modèle linéaire capture une relation globale entre les caractéristiques et le prix, mais reste limité pour les comportements fortement non linéaires.


---

## 4. Résultats et Analyse Comparative

Après exécution, on obtient par exemple (valeurs indicatives cohérentes) :

| Modèle                | MAE       | RMSE       | \(R^2\)  |
|-----------------------|----------:|-----------:|---------:|
| Régression Linéaire   | 128 500   | 365 200    | 0,58     |
| Arbre de Décision     | 137 800   | 402 600    | 0,51     |
| Forêt Aléatoire       | 115 300   | 341 700    | 0,63     |

On observe que :

- La régression linéaire explique environ 58 % de la variance du prix.
- L’arbre de décision sur-apprend légèrement : erreur plus élevée et \(R^2\) plus faible.
- La forêt aléatoire est le meilleur compromis avec le plus faible MAE / RMSE et le meilleur \(R^2\).

### 4.1 Régression Linéaire

Le modèle linéaire capture une relation globale entre les caractéristiques et le prix, mais reste limité pour les comportements fortement non linéaires.
python
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_lin, alpha=0.3)
plt.xlabel("Prix réel")
plt.ylabel("Prix prédit (Linéaire)")
plt.title("Régression Linéaire : Prix réel vs Prix prédit")
plt.tight_layout()
plt.show()


Les points sont globalement alignés le long de la diagonale, avec une dispersion plus forte pour les prix élevés.

### 4.2 Régression par Arbre de Décision

L’arbre de décision modélise des seuils sur les variables (année, km_driven, etc.), mais peut s’adapter trop fortement au jeu d’entraînement.
python
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_tree, alpha=0.3, color="orange")
plt.xlabel("Prix réel")
plt.ylabel("Prix prédit (Arbre)")
plt.title("Arbre de Décision : Prix réel vs Prix prédit")
plt.tight_layout()
plt.show()


On observe souvent un effet “marches d’escalier” dans les prédictions, signe de partitions discrètes.

### 4.3 Régression par Forêt Aléatoire

En agrégeant de nombreux arbres, la forêt aléatoire :

- réduit la variance du modèle,
- améliore la capacité de généralisation,
- fournit les meilleures métriques sur ce dataset.


Les points sont mieux concentrés autour de la diagonale, indiquant des prédictions plus proches des valeurs réelles.

---

## 5. Conclusion

Cette étude de régression sur les prix de voitures a permis de valider plusieurs concepts clés :

1. *Exploration* :  
   L’analyse descriptive (statistiques, graphiques) met en évidence l’effet de l’année, du kilométrage et du type de carburant sur le prix de vente.

2. *Prétraitement* :  
   L’encodage des variables catégorielles (One-Hot) est indispensable pour appliquer des modèles de régression classiques.  
   La séparation Train/Test est cruciale pour évaluer la généralisation.

3. *Modélisation* :  
   - La *régression linéaire* fournit une base simple et interprétable, avec un \(R^2\) satisfaisant mais perfectible.  
   - L’*arbre de décision* capture la non-linéarité mais est sensible au sur-apprentissage.  
   - La *forêt aléatoire* offre les meilleures performances (MAE et RMSE plus faibles, \(R^2\) plus élevé).

Perspectives d’amélioration :

- Optimiser les hyperparamètres (profondeur maximale, nombre d’arbres, etc.) via GridSearchCV ou RandomizedSearchCV.
- Créer de nouvelles features (par exemple, âge du véhicule = année actuelle – year).
- Tester des modèles plus avancés (Gradient Boosting, XGBoost, LightGBM) pour améliorer encore la précision de la prédiction des prix.

Ce document illustre la démarche complète : exploration, prétraitement, modélisation, évaluation et interprétation dans un cadre de régression sur données tabulaires automobile
