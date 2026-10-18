# Heavy Equipment Selling Price Prediction

Kaggle competition — predict auction sale prices of heavy construction equipment using classical ML.

**Leaderboard Score: 0.19727 RMSLE** | Cutoff: 0.2 

---

## Approach

- **EDA** — sales trends, depreciation curves, statistical tests (ANOVA, Pearson, Spearman)
- **Preprocessing** — sklearn Pipeline + ColumnTransformer, binary flags, label encoding, OHE
- **Feature Engineering** — AssetAge, HoursPerYear, ManufactureDecade, time features
- **Models** — Baseline → Linear Regression → Random Forest → XGBoost → CatBoost → LightGBM
- **Tuning** — Manual → RandomizedSearchCV → GridSearchCV → Extended GridSearch
- **Final Model** — 15-seed LightGBM ensemble, 71% improvement over baseline

## Results

| Model | Val RMSLE |
|---|---|
| Baseline | 0.6707 |
| LightGBM Base | 0.2229 |
| GridSearchCV | 0.2070 |
| **15-Seed Ensemble** | **0.1969** |

**Leaderboard: 0.19727**

## Stack
Python · Scikit-learn · LightGBM · XGBoost · CatBoost · Pandas · Matplotlib

