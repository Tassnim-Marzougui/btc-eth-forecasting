# btc-eth-forecasting

Prévision du Bitcoin et de l'Ethereum : ARIMA/SARIMA, VAR/VECM, GARCH, LSTM/GRU et analyse assistée par LLM, en Python et R.

Mini-projet Séries Temporelles — Tek-Up University (Data Science & IA), 2026.

## Données
- Bitcoin (BTC-USD) et Ethereum (ETH-USD), cours journaliers, Yahoo Finance
- Période : 01/01/2018 – 30/09/2026 (3 195 jours)
- Split chronologique 90 % train / 10 % test (`data/split.json`)

## Modèles
| Partie | Modèles | Langages |
|---|---|---|
| Analyse exploratoire + ARIMA/SARIMA | ADF/KPSS, STL, ARIMA, SARIMA | Python, R |
| VAR + Cointégration + GARCH | VAR, Granger, Johansen, VECM, GARCH | Python, R |
| Deep Learning + LLM | LSTM, GRU, analyse assistée par LLM | Python |

## Exécution
Python : `pip install yfinance statsmodels pmdarima pandas numpy matplotlib scipy`, puis ouvrir les notebooks dans l'ordre et « Run All ».

R : ouvrir les fichiers `.Rmd` dans RStudio, puis « Knit ».

## Équipe
- Tassnim Marzougui — analyse exploratoire, ARIMA/SARIMA
- Nourhen — VAR, cointégration, GARCH
- Oumaima — LSTM/GRU, LLM, comparaison finale