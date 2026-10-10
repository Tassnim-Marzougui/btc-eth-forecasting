# BTC & ETH — cointégration, volatilité et Deep Learning

Mini-projet Séries Temporelles — Tek-Up University (Data Science & IA), 2026.

**Question** : BTC et ETH partagent-ils un équilibre de long terme, et un modèle économétrique ou neuronal bat-il significativement la marche aléatoire pour prévoir le prix à 10 jours ?

**Réponse courte** : non. Aucun modèle (ARIMA, VAR, VECM, LSTM, GRU) ne bat significativement la marche aléatoire (test de Diebold-Mariano à 5 %). Il n'y a pas de cointégration BTC–ETH (Johansen et Engle-Granger), et le VECM est significativement moins bon que Naive. La volatilité, elle, est modélisable (GARCH(1,1)).

## Données
- Bitcoin (BTC-USD) et Ethereum (ETH-USD), cours journaliers, Yahoo Finance.
- Période : 01/01/2018 – 30/09/2026 (3 195 jours). **CSV figé dans `data/`** : ne pas re-télécharger (Yahoo peut corriger l'historique).
- Split chronologique 90 % / 10 % défini dans `data/split.json` : train jusqu'au 14/11/2025 (2 875 jours), test du 15/11/2025 au 30/09/2026 (320 jours).

## Protocole d'évaluation commun
- Rolling origin : une origine tous les 10 jours dans le test (**32 origines**), prévision à horizon h = 1 à 10.
- Erreurs mesurées sur le **prix en USD** (MSE, RMSE, MAE, MAPE).
- Baselines : marche aléatoire (Naive) et drift.
- Test de Diebold-Mariano (correction de Harvey) entre tous les modèles.
- Seed 42 pour les réseaux de neurones.

## Structure du dépôt
```
btc_eth/
├── data/                          btc_eth.csv, btc_eth_processed.csv, split.json
├── 01_data_eda.(ipynb|Rmd|html)   P1 — analyse exploratoire, ADF/KPSS, STL (Python + R)
├── 02_arima.(ipynb|Rmd|html)      P1 — ARIMA/SARIMA, diagnostics, prévisions (Python + R)
├── 03_var_coint.(py|R)            P2 — VAR, Granger, Johansen, VECM
├── 04_garch.(py|R)                P2 — GARCH, EGARCH/GJR, VaR, volatilité
├── src/deep_learning/
│   ├── 05_lstm_gru.py             P3 — LSTM/GRU (PyTorch), CV temporelle, walk-forward
│   ├── 06_llm_analysis.py         P3 — vérification des réponses du LLM contre nos sorties
│   └── 07_tableau_final.py        P3 — tableau final de tous les modèles + tests DM
├── notebooks/03_DeepLearning_LLM/ grok_reponses.md, interpretation_equipe.md (+ notebook Kaggle)
├── results/figures/               graphiques (01_ à 05_)
├── results/tables/                tableaux et prévisions (forecasts_<modele>_<actif>.csv)
├── report/                        rapport final
├── requirements.txt
└── README.md
```

## Exécuter le projet de bout en bout
Toutes les commandes se lancent **depuis la racine du dépôt** (les chemins sont relatifs).

1. Installer : `pip install -r requirements.txt`
2. **P1** : ouvrir `01_data_eda.ipynb` puis `02_arima.ipynb` (Run All). Version R : knit de `01_data_eda.Rmd` et `02_arima.Rmd` dans RStudio.
3. **P2** : `python 03_var_coint.py` puis `python 04_garch.py`. Version R : `Rscript 03_var_coint.R` et `Rscript 04_garch.R`.
4. **P3 — LSTM/GRU** : le script `src/deep_learning/05_lstm_gru.py` s'exécute sur **Kaggle avec GPU** (PyTorch). Ajouter le dépôt comme *Dataset*, coller le script dans un notebook, activer le GPU, *Run All* (`QUICK = True` pour un test rapide). Récupérer `results/` depuis l'onglet Output (prévisions `forecasts_lstm_*.csv`, `forecasts_gru_*.csv`, tableaux `05_*.csv`, figures `05_*.png`) et les placer dans `results/`.
5. **Tableau final et DM** : `python src/deep_learning/07_tableau_final.py` (produit `results/tables/07_*.csv`).
6. **Analyse du LLM** : `python src/deep_learning/06_llm_analysis.py --file notebooks/03_DeepLearning_LLM/grok_reponses.md` (produit `results/tables/06_*.csv`).

## Résultats clés (RMSE sur le prix, 32 origines, h = 1–10)
| Modèle | BTC | ETH |
|---|---|---|
| ARIMA (AIC) | 3 599 | 143,0 |
| Naive (= ARIMA BIC) | 3 607 | 145,0 |
| GRU | 3 684 | 145,3 |
| LSTM | 3 689 | 146,0 |
| VAR | 3 716 | 144,9 |
| VECM | 3 922 | 158,3 |

Détail complet : `results/tables/07_tableau_final.csv`, tests DM : `07_dm_vs_naive.csv` et `07_dm_matrice_pvalues.csv`. Le GARCH est évalué à part sur la volatilité (`04_metriques_garch.csv`).

## Limites
32 origines seulement (tests peu puissants), une seule période de test, réseaux univariés avec une seule graine, intervalles de confiance trop étroits à h = 10.

## Équipe
- Tassnim Marzougui — analyse exploratoire, ARIMA/SARIMA
- Nourhen — VAR, cointégration, GARCH
- Oumaima — LSTM/GRU, LLM, comparaison finale