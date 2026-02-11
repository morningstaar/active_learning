#  Active Learning: Sélection Intelligente de Données

## Vue d'ensemble

Ce projet implémente une pipeline d'**Active Learning** pour sélectionner intelligemment les données non annotées du **domaine B** qui vont améliorer le plus un modèle entraîné sur le **domaine A**.

### Pipeline global:
```
1. Modèle entraîné sur Domaine A (gaze_model_trained.pth)
   ↓
2. Évaluation sur Domaine B (baseline)
   ↓
3. Sélection intelligente de 50 données via 3 stratégies
   ↓
4. [Prochaine étape] Annotation manuelle + ré-entraînement
   ↓
5. [Prochaine étape] Mesure d'amélioration
```

---

##  Résultats Baseline (Domaine B sans adaptation)

**Performance du checkpoint sur le testSet du domaine B:**
- **MSE:** 0.064411
- **MAE:** 0.208175 (**≈ 206.5 pixels**) 
- **RMSE:** 0.254045

**Interprétation:** Le modèle se trompe en moyenne de **206 pixels** sur les données du domaine B. C'est loin d'être optimal - d'où la nécessité d'adapter le modèle.

---

##  Les 3 Stratégies d'Active Learning Implémentées

### **Stratégie 1: Entropy Sampling**  **INCERTITUDE**

#### Qu'est-ce que c'est?
Sélectionne les données où le modèle **hésite le plus**.

#### Fonctionnement:
```python
# Pour chaque prédiction [x_pred, y_pred] en [0, 1]
uncertainty = |x_pred - 0.5| + |y_pred - 0.5|
# Plus proche de 0.5 = plus incertain
```

#### Exemple:
-  Prédiction confiante: `x_pred=0.95, y_pred=0.92` → loin de 0.5 → FAIBLE incertitude
-  Prédiction hésitante: `x_pred=0.51, y_pred=0.48` → proche de 0.5 → FORTE incertitude

#### Pourquoi?
**"Si le modèle n'est pas sûr, il a besoin d'exemples pour apprendre !"**

Quand le modèle prédit `0.5` (50-50), il n'a aucune confiance. Ces cas-là sont les plus utiles pour le ré-entraînement.

---

### **Stratégie 2: Margin Sampling**  **INCERTITUDE (variante)**

#### Qu'est-ce que c'est?
Variante de l'Entropy Sampling - sélectionne les données **proches de la limite de décision**.

#### Fonctionnement:
```python
# Marge de confiance
margin = (|x_pred - 0.5| + |y_pred - 0.5|) / 2

# Sélectionne les K petites marges (< prédictions confiantes)
```

#### Différence avec Entropy:
- **Entropy:** Distance à 0.5 → regarde chaque coordonnée
- **Margin:** Distance moyenne à 0.5 → regarde le couple (x, y)

#### Pourquoi?
**"Les décisions au bord de la falaise sont instables - ce sont les cas où l'apprentissage compte !"**

C'est une deuxième opinion pour confirmer quelles données sont vraiment incertaines.

---

### **Stratégie 3: Diversity Sampling**  **DIVERSITÉ**

#### Qu'est-ce que c'est?
Sélectionne des données qui **représentent différentes zones** de l'espace des features du modèle.

#### Fonctionnement (détaillé):

```python
# Step 1: Extraire les features internes du modèle
# Pour chaque image, on capture la sortie de la dernière couche cachée:
#   features = ReLU(fc1(flatten(conv2(pool(conv1(image))))))
# → Vecteur de 256 dimensions par image

# Step 2: Clustering K-means (5 clusters)
#   Regroupe les 7149 images en 5 groupes similaires
#   Chaque cluster = une "région" de l'espace des features

# Step 3: Calculer distance au centre
#   Pour chaque image: distance(features_image, centre_cluster)
#   Plus loin du centre = plus "extrême"

# Step 4: Sélectionner les plus éloignées
#   Top 50 images les plus éloignées de leur centre de cluster
#   → Ce sont les images les plus "représentatives de la diversité"
```

#### Visualisation:
```
Cluster 1 (ex: "regard haut-gauche")
    •
   • •  ← Image sélectionnée (loin du centre)
  •
 •X•    ← Centre du cluster (images similaires)
  •

Cluster 2 (ex: "regard bas-droit")
         •
    •   •  ← Image sélectionnée
       •
    •X••   ← Centre du cluster
```

#### Pourquoi?
**"Je ne veux pas d'exemples redondants - je veux couvrir TOUS les cas possibles !"**

-  Si on sélectionne 50 images identiques → pas utile
-  Si on sélectionne 50 images très différentes → force le modèle à généraliser

---

##  Sélection Finale: Fusion des 3 Stratégies (Voting)

### Processus:
```python
entropy_top_50 = [idx1, idx2, ..., idx50]      # Les 50 plus incertaines
margin_top_50   = [idx3, idx4, ..., idx52]     # Les 50 près de la limite
diversity_top_50 = [idx5, idx6, ..., idx54]    # Les 50 plus diversifiées

# Union: prendre les indices qui apparaissent dans au moins une liste
selected_ensemble = entropy_top_50 ∪ margin_top_50 ∪ diversity_top_50
# Limiter à 50 pour l'annotation manuelle
final_selection = selected_ensemble[:50]
```

### Résultat:
 **50 données sélectionnées** qui combinent:
- Images où le modèle **doute**
- Images qui **varient beaucoup** en termes de features internes
- Couverture **maximale** de l'espace des données

---

## 📁 Fichiers Générés

### `selected_for_annotation.csv`
```
frame_id,timestamp,x,y,selected
519,12.34,1250,612,True
520,12.37,1261,615,True
521,12.40,1273,618,True
...
```

**Colonnes:**
- `frame_id`: ID de la frame
- `timestamp`: Temps dans la vidéo
- `x, y`: Coordonnées du regard en pixels
- `selected`: `True` (marquée pour annotation)

**Utilisation:**
- Pour annoter manuellement ou vérifier les annotations
- Pour le ré-entraînement

---

##  Comparaison des Stratégies

| Stratégie | Focus | Cas d'usage | Coût computationnel |
|-----------|-------|----------|---|
| **Entropy** | "Où tu hésites?" | Quick wins - erreurs faciles | ⭐ Faible |
| **Margin** | "Tu es sûr?" | Confirmation d'incertitude | ⭐ Faible |
| **Diversity** | "Couvre tout?" | Généralisation complète | ⭐⭐ Moyen (K-means) |
| **Ensemble** | "Tout combiner" | Meilleure couverture | ⭐⭐ Moyen |

---

##  Prochaines Étapes

1. **Annotation manuelle** des 50 données sélectionnées
   - Corriger ou valider les labels (x, y)
   - Sauvegarder dans `selected_for_annotation.csv`

2. **Ré-entraînement**
   - Combiner: Domaine A (entraînement original) + 50 nouvelles données
   - Fine-tune le checkpoint sur ce nouvel ensemble
   - Époque(s) supplémentaires

3. **Évaluation**
   - Tester le nouveau modèle sur testSet Domaine B
   - Comparer MAE/RMSE avec la baseline
   - Mesurer l'amélioration: `(Baseline - Nouveau) / Baseline × 100%`

4. **Itération [optionnel]**
   - Si amélioration insuffisante
   - Sélectionner 50 autres données incertaines
   - Répéter

---

##  Intuition Globale

### Pourquoi Active Learning?

**Sans Active Learning:**
- Annoter au hasard: `selected_indices = random(0, 7149, 50)`
- Résultat: Peut tomber sur des données faciles ou redondantes
- Ré-entraînement inefficace

**Avec Active Learning (notre approche):**
- Sélectionner **intelligemment**: `entropy + margin + diversity`
- Résultat: Les 50 données **les plus utiles**
- Ré-entraînement **bien plus efficace**

### Budget:
- 50 annotations à faire (humainement faisable)
- vs 7149 annotations complètes (non faisable)
- **Ratio:** 50/7149 = **0.7%** du coût total !

---

##  Implémentation Technique

### Framework:
- **PyTorch** pour le modèle
- **scikit-learn** pour K-means (Diversity)
- **pandas** pour gérer les données
- **OpenCV** pour charger les images

### Architecture du modèle:
```
GazeNet (entraîné sur Domaine A)
├── Conv1 (3 → 16 channels)
├── MaxPool (4x4)
├── Conv2 (16 → 32 channels)
├── MaxPool (4x4)
├── FC1 (6272 → 256)      ← Features extraites pour Diversity
├── ReLU
└── FC2 (256 → 2)         ← Sortie: [x, y] normalisés
```

### Performance (Baseline Domaine B):
- **Chargement Domaine B:** 7149 images
- **Évaluation Baseline:** ~60s
- **Sélection (3 stratégies):** ~400s
- **Sauvegarde:** instantanée

---

##  Références

- **Uncertainty Sampling:** Freeman, 1965; Lewis & Gale, 1994
- **Diversity Sampling:** Dagan & Engelson, 1995
- **Active Learning Survey:** Settles, 2012
- **Monte Carlo Dropout:** Gal & Ghahramani, 2016 (non implémenté ici)

---

##  Checklist

- [x] Charger modèle entraîné (Domaine A)
- [x] Charger données non annotées (Domaine B)
- [x] Implémenter Entropy Sampling
- [x] Implémenter Margin Sampling
- [x] Implémenter Diversity Sampling
- [x] Fusion par ensemble voting
- [x] Visualiser 6 exemples sélectionnés
- [x] Sauvegarder indices → `selected_for_annotation.csv`
- [ ] **Annotation manuelle des 50 données**
- [ ] **Ré-entraînement avec nouvelles données**
- [ ] **Évaluation et mesure d'amélioration**

---

##  Auteur & Date

- **Créé:** Février 2026
- **Projet:** Active Learning pour Gaze Tracking (Cross-Domain Adaptation)
- **Domaine A:** Source domain (entraînement initial)
- **Domaine B:** Target domain (adaptation)
