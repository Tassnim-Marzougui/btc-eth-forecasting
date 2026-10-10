"""
03_var_coint.py : Personne 2
Relation BTC-ETH, stationnarite, cointegration (Engle-Granger / Johansen),
VAR, causalite de Granger, VECM, IRF/FEVD et evaluation walk-forward EN PRIX.

Sorties :
  results/tables/forecasts_P2.csv        (format commun pour le test Diebold-Mariano)
  results/tables/03_metriques_var.csv    (RMSE/MAE/MAPE en USD)
  results/tables/03_metriques_par_horizon.csv
  + tables et figures intermediaires
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import het_arch

warnings.filterwarnings('ignore')

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

# ----------------------------------------------------------------------
# Parametres du protocole commun (a aligner avec P1 et P3)
# ----------------------------------------------------------------------
H = 10            # horizon de prevision (pas)
N_ORIGINS = 32    # nombre d'origines walk-forward
ORIGINS_FILE = 'data/origins.json'   # liste commune d'origines (positions entières) si fournie par P1

# ======================================================================
print('=' * 60)
print('SECTION 1: Data Loading & Preparation')
print('=' * 60)
# ======================================================================
df = pd.read_csv('data/btc_eth_processed.csv', parse_dates=['date'], index_col='date')
with open('data/split.json', 'r') as f:
    sp = json.load(f)

train = df.loc[:sp['train_end']]
test = df.loc[sp['test_start']:sp['test_end']]

print(f"Train size: {len(train)}")
print(f"Test size: {len(test)}")

# ======================================================================
print('\n' + '=' * 60)
print('SECTION 2: BTC-ETH Relationship Study')
print('=' * 60)
# ======================================================================
corr_logp = train[['btc_logp', 'eth_logp']].corr().iloc[0, 1]
corr_ret = train[['btc_ret', 'eth_ret']].dropna().corr().iloc[0, 1]
print(f"Correlation de Pearson (log prix): {corr_logp:.4f}")
print(f"Correlation de Pearson (rendements): {corr_ret:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(train['btc_logp'], train['eth_logp'], alpha=0.5)
plt.title('Scatter plot: BTC log price vs ETH log price')
plt.xlabel('BTC Log Price')
plt.ylabel('ETH Log Price')
plt.savefig('results/figures/03_scatter_logp.png', dpi=150, bbox_inches='tight')
plt.close()

rolling_corr_logp = train['btc_logp'].rolling(window=90).corr(train['eth_logp'])
rolling_corr_ret = train['btc_ret'].rolling(window=90).corr(train['eth_ret'])

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
rolling_corr_logp.plot(ax=axes[0])
axes[0].set_title('Correlation glissante 90 jours (log prix)')
axes[0].set_ylabel('Correlation')
rolling_corr_ret.plot(ax=axes[1])
axes[1].set_title('Correlation glissante 90 jours (rendements)')
axes[1].set_ylabel('Correlation')
plt.tight_layout()
plt.savefig('results/figures/03_correlation_glissante.png', dpi=150, bbox_inches='tight')
plt.close()

pd.DataFrame({'Pearson_logp': [corr_logp], 'Pearson_ret': [corr_ret]}).to_csv(
    'results/tables/03_correlation.csv', index=False)

# ======================================================================
print('\n' + '=' * 60)
print('SECTION 3: Stationarity Tests for Multivariate Analysis')
print('=' * 60)
# ======================================================================
def run_stationarity_tests(series, name):
    s = series.dropna()
    adf_result = adfuller(s)
    kpss_result = kpss(s, regression='c')
    return {
        'Variable': name,
        'ADF_Stat': adf_result[0],
        'ADF_p_value': adf_result[1],
        'KPSS_Stat': kpss_result[0],
        'KPSS_p_value': kpss_result[1],
    }

variables = {
    'btc_logp': train['btc_logp'],
    'eth_logp': train['eth_logp'],
    'btc_ret': train['btc_ret'],
    'eth_ret': train['eth_ret'],
    'btc_logp_diff': train['btc_logp'].diff(),
    'eth_logp_diff': train['eth_logp'].diff(),
}
stat_df = pd.DataFrame([run_stationarity_tests(s, n) for n, s in variables.items()])
stat_df.to_csv('results/tables/03_stationnarite_multi.csv', index=False)
print(stat_df)

# ======================================================================
print('\n' + '=' * 60)
print('SECTION 4: Cointegration Tests')
print('=' * 60)
# ======================================================================
print("--- Engle-Granger Test ---")
eg_score, eg_pval, _ = coint(train['btc_logp'].dropna(), train['eth_logp'].dropna())
print(f"Engle-Granger stat: {eg_score:.4f} | p-value: {eg_pval:.4f}")
pd.DataFrame({'EG_stat': [eg_score], 'EG_p_value': [eg_pval]}).to_csv(
    'results/tables/03_engle_granger.csv', index=False)

print("\n--- Johansen Test ---")
data_johansen = train[['btc_logp', 'eth_logp']].dropna()
johansen_res = coint_johansen(data_johansen, det_order=0, k_ar_diff=2)

johansen_df = pd.DataFrame({
    'Trace_Stat': johansen_res.lr1,
    'Trace_Crit_90%': johansen_res.cvt[:, 0],
    'Trace_Crit_95%': johansen_res.cvt[:, 1],
    'Trace_Crit_99%': johansen_res.cvt[:, 2],
    'Max_Eig_Stat': johansen_res.lr2,
    'Max_Eig_Crit_90%': johansen_res.cvm[:, 0],
    'Max_Eig_Crit_95%': johansen_res.cvm[:, 1],
    'Max_Eig_Crit_99%': johansen_res.cvm[:, 2],
}, index=['r=0', 'r<=1'])
print(johansen_df)
johansen_df.to_csv('results/tables/03_cointegration.csv')

# Rang de cointegration retenu (trace test, 95 %), traite sequentiellement
coint_rank = 0
for i in range(len(johansen_res.lr1)):
    if johansen_res.lr1[i] > johansen_res.cvt[i, 1]:
        coint_rank += 1
    else:
        break
print(f"\nRang de cointegration (Johansen, trace, 95%): {coint_rank}")
if coint_rank == 0:
    print("ATTENTION : aucune cointegration detectee par Johansen. "
          "Le VECM (rang 1) est conserve a titre de comparaison uniquement ; "
          "le signaler dans le rapport.")
vecm_rank = max(coint_rank, 1)

# ======================================================================
print('\n' + '=' * 60)
print('SECTION 5: VAR Model')
print('=' * 60)
# ======================================================================
data_var = train[['btc_ret', 'eth_ret']].dropna()
model = VAR(data_var)

lag_selection = model.select_order(maxlags=20)
print(lag_selection.summary())

lag_df = pd.DataFrame({
    'Critere': ['AIC', 'BIC', 'FPE', 'HQIC'],
    'Retard_Optimal': [lag_selection.aic, lag_selection.bic,
                       lag_selection.fpe, lag_selection.hqic],
})
lag_df.to_csv('results/tables/03_var_selection_retards.csv', index=False)

opt_lag = max(lag_selection.aic, 1)  # au moins 1 retard
print(f"\nOrdre de retard optimal choisi (AIC): {opt_lag}")

var_res = model.fit(opt_lag)
print(var_res.summary())

# Diagnostic des residus : ARCH-LM (justifie le GARCH de 04_garch)
arch_rows = []
for col in var_res.resid.columns:
    lm_stat, lm_p, f_stat, f_p = het_arch(var_res.resid[col], nlags=10)
    arch_rows.append({'Serie': col, 'LM_stat': lm_stat, 'LM_p_value': lm_p,
                      'F_stat': f_stat, 'F_p_value': f_p})
arch_df = pd.DataFrame(arch_rows)
arch_df.to_csv('results/tables/03_arch_lm_var.csv', index=False)
print("\nARCH-LM sur les residus du VAR:")
print(arch_df)

# ======================================================================
print('\n' + '=' * 60)
print('SECTION 6: Granger Causality Tests')
print('=' * 60)
# ======================================================================
granger_btc_eth = var_res.test_causality('eth_ret', 'btc_ret', kind='f')
granger_eth_btc = var_res.test_causality('btc_ret', 'eth_ret', kind='f')

print("H0: BTC ne cause pas ETH au sens de Granger")
print(f"p-value: {granger_btc_eth.pvalue:.4f}")
print("\nH0: ETH ne cause pas BTC au sens de Granger")
print(f"p-value: {granger_eth_btc.pvalue:.4f}")

pd.DataFrame({
    'Direction': ['BTC -> ETH', 'ETH -> BTC'],
    'Test_Stat': [granger_btc_eth.test_statistic, granger_eth_btc.test_statistic],
    'p_value': [granger_btc_eth.pvalue, granger_eth_btc.pvalue],
}).to_csv('results/tables/03_granger.csv', index=False)

# ======================================================================
print('\n' + '=' * 60)
print('SECTION 7: VECM')
print('=' * 60)
# ======================================================================
# VAR(p) sur rendements <=> VECM avec k_ar_diff = p
vecm_model = VECM(data_johansen, k_ar_diff=opt_lag, coint_rank=vecm_rank, deterministic='co')
vecm_res = vecm_model.fit()
print(vecm_res.summary())

with open('results/tables/03_vecm_summary.txt', 'w') as f:
    f.write(str(vecm_res.summary()))

alpha_btc = vecm_res.alpha[0, 0]
alpha_eth = vecm_res.alpha[1, 0]
beta = vecm_res.beta[:, 0]
print("\nVecteur de cointegration (beta, normalise sur BTC):")
print(f"  btc_logp: {beta[0]:.4f} | eth_logp: {beta[1]:.4f}")
print("Termes de correction d'erreur (alpha):")
print(f"  BTC: {alpha_btc:.4f}")
print(f"  ETH: {alpha_eth:.4f}")
print("Interpretation : alpha < 0 => la variable corrige l'ecart a l'equilibre ; "
      "|alpha| mesure la vitesse d'ajustement par periode ; "
      "alpha ~ 0 => variable faiblement exogene.")

pd.DataFrame({
    'Variable': ['btc_logp', 'eth_logp'],
    'alpha': [alpha_btc, alpha_eth],
    'beta': [beta[0], beta[1]],
}).to_csv('results/tables/03_vecm_alpha_beta.csv', index=False)

# ======================================================================
print('\n' + '=' * 60)
print('SECTION 8: Walk-forward en PRIX (origines x h=1..%d)' % H)
print('=' * 60)
# ======================================================================
# Tous les modeles sont convertis en prix (USD) pour etre comparables a
# ARIMA (P1) et LSTM/GRU (P3) et pour le test de Diebold-Mariano.
full = df[['btc_logp', 'eth_logp']].dropna()

if os.path.exists(ORIGINS_FILE):
    with open(ORIGINS_FILE, 'r') as f:
        origins = np.array(json.load(f), dtype=int)
    print(f"Origines chargees depuis {ORIGINS_FILE}: {len(origins)}")
else:
    test_start_pos = full.index.get_loc(pd.Timestamp(sp['test_start']))
    last_origin = len(full) - H
    origins = np.linspace(test_start_pos, last_origin, N_ORIGINS).astype(int)
    print(f"Origines generees ({len(origins)}) : A REMPLACER par la liste commune de P1/P3 "
          f"(data/origins.json).")

assets = ['BTC', 'ETH']
rows = []

for t in origins:
    train_t = full.iloc[:t]
    P_t = np.exp(train_t.iloc[-1].values)                  # derniers prix connus (2,)
    true = np.exp(full.iloc[t:t + H].values)               # prix reels (H, 2)
    dates = full.index[t:t + H]
    if len(true) < H:
        continue

    # --- VAR sur rendements -> prix
    ret_t = train_t.diff().dropna()
    var_t = VAR(ret_t).fit(opt_lag)
    r_hat = var_t.forecast(ret_t.values[-opt_lag:], steps=H)
    pred_var = P_t * np.exp(np.cumsum(r_hat, axis=0))

    # --- VECM sur log-prix -> prix
    vecm_t = VECM(train_t, k_ar_diff=opt_lag, coint_rank=vecm_rank, deterministic='co').fit()
    raw = vecm_t.predict(steps=H)
    raw = raw[0] if isinstance(raw, tuple) else raw
    pred_vecm = np.exp(raw)

    # --- Reference : marche aleatoire (dernier prix)
    pred_rw = np.tile(P_t, (H, 1))

    for name, pred in [('VAR', pred_var), ('VECM', pred_vecm), ('RW', pred_rw)]:
        for h in range(H):
            for j, a in enumerate(assets):
                rows.append((int(t), dates[h], h + 1, a, name, true[h, j], pred[h, j]))

fc = pd.DataFrame(rows, columns=['origin', 'date', 'h', 'asset', 'model', 'y_true', 'y_pred'])
fc.to_csv('results/tables/forecasts_P2.csv', index=False)
print(f"Prevision exportee : results/tables/forecasts_P2.csv ({len(fc)} lignes)")

def metrics(g):
    e = g['y_true'] - g['y_pred']
    return pd.Series({
        'RMSE': np.sqrt((e ** 2).mean()),
        'MAE': e.abs().mean(),
        'MAPE': (e.abs() / g['y_true']).mean() * 100,
    })

metrics_df = fc.groupby(['model', 'asset']).apply(metrics).reset_index()
metrics_df.to_csv('results/tables/03_metriques_var.csv', index=False)
print("\nMetriques globales (prix, USD):")
print(metrics_df)

by_h = fc.groupby(['model', 'asset', 'h']).apply(metrics).reset_index()
by_h.to_csv('results/tables/03_metriques_par_horizon.csv', index=False)

# Graphiques : une origine au milieu, BTC et ETH
t0 = origins[len(origins) // 2]
for a in assets:
    g = fc[(fc.origin == t0) & (fc.asset == a)]
    fig, ax = plt.subplots(figsize=(12, 6))
    ref = g[g.model == 'VAR']
    ax.plot(ref['date'], ref['y_true'], 'k-', label='Reel')
    for m in ['VAR', 'VECM', 'RW']:
        gm = g[g.model == m]
        ax.plot(gm['date'], gm['y_pred'], '--', label=m)
    ax.set_title(f'{a} : previsions en prix a {H} pas (origine {t0})')
    ax.set_ylabel('Prix (USD)')
    ax.legend()
    plt.savefig(f'results/figures/03_previsions_prix_{a.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()

# RMSE par horizon
fig, ax = plt.subplots(figsize=(10, 6))
for m in ['VAR', 'VECM', 'RW']:
    s = by_h[(by_h.model == m) & (by_h.asset == 'BTC')]
    ax.plot(s['h'], s['RMSE'], marker='o', label=m)
ax.set_title('RMSE par horizon (BTC, prix)')
ax.set_xlabel('Horizon h')
ax.set_ylabel('RMSE (USD)')
ax.legend()
plt.savefig('results/figures/03_rmse_par_horizon_btc.png', dpi=150, bbox_inches='tight')
plt.close()

# ======================================================================
print('\n' + '=' * 60)
print('SECTION 9: IRF (Impulse Response Functions)')
print('=' * 60)
# ======================================================================
irf = var_res.irf(20)
irf.plot()
plt.savefig('results/figures/03_irf.png', dpi=150, bbox_inches='tight')
plt.close()

# ======================================================================
print('\n' + '=' * 60)
print('SECTION 10: FEVD (Forecast Error Variance Decomposition)')
print('=' * 60)
# ======================================================================
fevd = var_res.fevd(20)
fevd.plot()
plt.savefig('results/figures/03_fevd.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAnalyse VAR/VECM terminee avec succes !")