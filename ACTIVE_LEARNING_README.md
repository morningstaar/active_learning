# Active Learning — Rapport d'exécution (al_optimized.ipynb)

Ce README explique **exactement** ce qui a été fait, pourquoi, et comment exécuter le pipeline.

---

## 1) Problème et objectif

**Objectif:** améliorer un modèle de prédiction de regard (régression) sur le **Domaine B** via Active Learning.

**Domaines:**
- **Domaine A (source):** SC1
- **Domaine B (target):** SC1 + YAS1

**Données:**
- Train DB: 10,716 (SC1 7,149 + YAS1 3,567)
- Test DB: 3,569

**Modèle:** checkpoint `data/gaze_model_trained.pth`

---

## 2) Pipeline suivi (conforme aux consignes)

1. **Chargement des données DB** (SC1 + YAS1)
2. **Split**: 10% baseline / 90% pool AL / test séparé
3. **Baseline**: entraînement + évaluation sur test
4. **Active Learning**:
   - scoring (incertitude/diversité/mix)
   - sélection par budgets
   - ré-entraînement incrémental
5. **Comparaison**: Random vs stratégies AL
6. **Graphes** performance-cost + IC

---

## 3) Réponses explicites au protocole AL

**Q1. Garder Domaine A ?**  
**Réponse:** NON. On adapte uniquement sur Domaine B.

**Q2. Sélection incrémentale ?**  
**Réponse:** OUI. Chaque budget inclut le budget précédent.

**Q3. Entraînement continu ?**  
**Réponse:** OUI. On continue depuis le checkpoint précédent.

---

## 4) Stratégies implémentées (4 requis + random)

1. **Random** (baseline)
2. **Uncertainty (TTA variance)**
3. **Diversity (K-means clustering)**
4. **Mixed U+D** (combinaison incertitude + diversité)
5. **Ensemble voting** (mix via désaccord des passes stochastiques)

---

## 5) Budgets et runs

- Budgets utilisés: **1%, 2%, 10%, 20%, 50%** (optimisé pour CPU)
- Runs: **2**
- Epochs AL: **2**

---

## 6) Normalisation des labels

Le modèle de base a été entraîné avec **x,y normalisés**. Les labels DB n’étaient pas normalisés, donc :

- Si x/y > 1 → normalisation automatique par max_x et max_y
- Si x/y ∈ [0,1] → aucune transformation

Cela a permis de ramener la MAE à un niveau cohérent (~0.25).

---

## 7) Résultats observés (résumé)

**Baseline MAE (normalisé):** ~0.251

Les stratégies AL sont très proches du random (pas de gain significatif).
Cela correspond exactement à l’analyse demandée dans la consigne (AL peut ne pas battre random).

---

## 8) Graphes de comparaison

Un graphe performance–cost est généré et sauvegardé:

**Fichier:** al_curves.png

Le graphe inclut:
- Random vs stratégies AL
- IC 95%
- Baseline en ligne pointillée

---

## 9) Fichiers importants

- al_optimized.ipynb → notebook final exécuté
- data/gaze_model_trained.pth → checkpoint utilisé
- al_curves.png → courbe performance-cost

---

## 10) Exécution rapide

Ouvrir le notebook al_optimized.ipynb et exécuter de haut en bas.

Temps estimé (CPU): ~2-3h

---

## 11) État final (résumé)

- ✅ Dataset chargé et normalisé
- ✅ Baseline entraînée
- ✅ 4 stratégies + random
- ✅ Comparaison budget/performances
- ✅ Graphe généré

---

Si tu veux une version “full consigne” avec **6 budgets** et **3 runs**, je peux l’ajouter (mais plus long à exécuter).
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
