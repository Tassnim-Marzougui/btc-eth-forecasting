import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM, select_order
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os
import json
import warnings

# Ignorer les avertissements
warnings.filterwarnings('ignore')

# Créer les répertoires si nécessaire
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

print('='*60)
print('SECTION 1: Data Loading & Preparation')
print('='*60)

# Charger les données
df = pd.read_csv('data/btc_eth_processed.csv', parse_dates=['date'], index_col='date')
with open('data/split.json', 'r') as f:
    sp = json.load(f)

train = df.loc[:sp['train_end']]
test = df.loc[sp['test_start']:sp['test_end']]

print(f"Train size: {len(train)}")
print(f"Test size: {len(test)}")

print('\n' + '='*60)
print('SECTION 2: BTC-ETH Relationship Study')
print('='*60)

# Corrélation de Pearson
corr = train[['btc_logp', 'eth_logp']].corr().iloc[0, 1]
print(f"Corrélation de Pearson entre BTC et ETH (log prix): {corr:.4f}")

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(train['btc_logp'], train['eth_logp'], alpha=0.5)
plt.title('Scatter plot: BTC log price vs ETH log price')
plt.xlabel('BTC Log Price')
plt.ylabel('ETH Log Price')
plt.savefig('results/figures/03_scatter_logp.png', dpi=150, bbox_inches='tight')
plt.close()

# Corrélation glissante
rolling_corr = train['btc_logp'].rolling(window=90).corr(train['eth_logp'])
plt.figure(figsize=(12, 6))
rolling_corr.plot()
plt.title('Corrélation glissante sur 90 jours (log prix)')
plt.ylabel('Corrélation')
plt.savefig('results/figures/03_correlation_glissante.png', dpi=150, bbox_inches='tight')
plt.close()

# Sauvegarder les stats de corrélation
corr_df = pd.DataFrame({'Pearson_Corr': [corr]})
corr_df.to_csv('results/tables/03_correlation.csv', index=False)

print('\n' + '='*60)
print('SECTION 3: Stationarity Tests for Multivariate Analysis')
print('='*60)

def run_stationarity_tests(series, name):
    # ADF Test
    adf_result = adfuller(series.dropna())
    # KPSS Test
    kpss_result = kpss(series.dropna(), regression='c')
    
    return {
        'Variable': name,
        'ADF_Stat': adf_result[0],
        'ADF_p_value': adf_result[1],
        'KPSS_Stat': kpss_result[0],
        'KPSS_p_value': kpss_result[1]
    }

variables = {
    'btc_logp': train['btc_logp'],
    'eth_logp': train['eth_logp'],
    'btc_ret': train['btc_ret'],
    'eth_ret': train['eth_ret'],
    'btc_logp_diff': train['btc_logp'].diff(),
    'eth_logp_diff': train['eth_logp'].diff()
}

stat_results = []
for name, series in variables.items():
    stat_results.append(run_stationarity_tests(series, name))

stat_df = pd.DataFrame(stat_results)
stat_df.to_csv('results/tables/03_stationnarite_multi.csv', index=False)
print(stat_df)

print('\n' + '='*60)
print('SECTION 4: Cointegration Tests')
print('='*60)

# 4a. Engle-Granger test
print("--- Engle-Granger Test ---")
score, pval, _ = coint(train['btc_logp'].dropna(), train['eth_logp'].dropna())
print(f"Engle-Granger p-value: {pval:.4f}")

# 4b. Johansen test
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
    'Max_Eig_Crit_99%': johansen_res.cvm[:, 2]
}, index=['r=0', 'r<=1'])

print(johansen_df)
johansen_df.to_csv('results/tables/03_cointegration.csv')

print('\n' + '='*60)
print('SECTION 5: VAR Model')
print('='*60)

data_var = train[['btc_ret', 'eth_ret']].dropna()
model = VAR(data_var)

# Sélection de l'ordre
lag_selection = model.select_order(maxlags=20)
print(lag_selection.summary())

lag_df = pd.DataFrame({
    'Critere': ['AIC', 'BIC', 'FPE', 'HQIC'],
    'Retard_Optimal': [lag_selection.aic, lag_selection.bic, lag_selection.fpe, lag_selection.hqic]
})
lag_df.to_csv('results/tables/03_var_selection_retards.csv', index=False)

opt_lag = max(lag_selection.aic, 1)  # Au moins 1 retard
print(f"\nOrdre de retard optimal choisi (AIC): {opt_lag}")

var_res = model.fit(opt_lag)
print(var_res.summary())

print('\n' + '='*60)
print('SECTION 6: Granger Causality Tests')
print('='*60)

granger_btc_eth = var_res.test_causality('eth_ret', 'btc_ret', kind='f')
granger_eth_btc = var_res.test_causality('btc_ret', 'eth_ret', kind='f')

print("H0: BTC ne cause pas ETH au sens de Granger")
print(f"p-value: {granger_btc_eth.pvalue:.4f}")

print("\nH0: ETH ne cause pas BTC au sens de Granger")
print(f"p-value: {granger_eth_btc.pvalue:.4f}")

granger_df = pd.DataFrame({
    'Direction': ['BTC -> ETH', 'ETH -> BTC'],
    'Test_Stat': [granger_btc_eth.test_statistic, granger_eth_btc.test_statistic],
    'p_value': [granger_btc_eth.pvalue, granger_eth_btc.pvalue]
})
granger_df.to_csv('results/tables/03_granger.csv', index=False)

print('\n' + '='*60)
print('SECTION 7: VECM (if cointegration detected)')
print('='*60)

vecm_model = VECM(data_johansen, k_ar_diff=opt_lag-1 if opt_lag > 1 else 1, coint_rank=1, deterministic='co')
vecm_res = vecm_model.fit()
print(vecm_res.summary())

with open('results/tables/03_vecm_summary.csv', 'w') as f:
    f.write(vecm_res.summary().as_csv())

alpha_btc = vecm_res.alpha[0, 0]
alpha_eth = vecm_res.alpha[1, 0]
print(f"\nTermes de correction d'erreur (Alpha):")
print(f"BTC: {alpha_btc:.4f}")
print(f"ETH: {alpha_eth:.4f}")

print('\n' + '='*60)
print('SECTION 8: Forecasting & Evaluation')
print('='*60)

# VAR Forecasting sur test set
# Prévisions dynamiques pour simuler le comportement du modèle
steps = len(test)
lagged_values = data_var.values[-opt_lag:]
var_forecast = var_res.forecast(y=lagged_values, steps=steps)
var_forecast_df = pd.DataFrame(var_forecast, index=test.index, columns=['btc_ret', 'eth_ret'])

test_var = test[['btc_ret', 'eth_ret']].dropna()

var_metrics = []
for asset in ['btc_ret', 'eth_ret']:
    rmse = np.sqrt(mean_squared_error(test_var[asset], var_forecast_df.loc[test_var.index, asset]))
    mae = mean_absolute_error(test_var[asset], var_forecast_df.loc[test_var.index, asset])
    mape = mean_absolute_percentage_error(test_var[asset], var_forecast_df.loc[test_var.index, asset])
    var_metrics.append({'Model': 'VAR', 'Asset': asset, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

# VECM Forecasting
vecm_forecast_raw = vecm_res.predict(steps=steps)
# predict() peut retourner un tuple (forecast, lower, upper) ou juste forecast
if isinstance(vecm_forecast_raw, tuple):
    vecm_forecast = vecm_forecast_raw[0]
else:
    vecm_forecast = vecm_forecast_raw
vecm_forecast_df = pd.DataFrame(vecm_forecast, index=test.index, columns=['btc_logp', 'eth_logp'])

test_vecm = test[['btc_logp', 'eth_logp']].dropna()

for asset in ['btc_logp', 'eth_logp']:
    rmse = np.sqrt(mean_squared_error(test_vecm[asset], vecm_forecast_df.loc[test_vecm.index, asset]))
    mae = mean_absolute_error(test_vecm[asset], vecm_forecast_df.loc[test_vecm.index, asset])
    mape = mean_absolute_percentage_error(test_vecm[asset], vecm_forecast_df.loc[test_vecm.index, asset])
    var_metrics.append({'Model': 'VECM', 'Asset': asset, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

metrics_df = pd.DataFrame(var_metrics)
metrics_df.to_csv('results/tables/03_metriques_var.csv', index=False)
print(metrics_df)

# Sauvegarder les prévisions
var_forecast_df[['btc_ret']].to_csv('results/tables/forecasts_var_btc.csv')
var_forecast_df[['eth_ret']].to_csv('results/tables/forecasts_var_eth.csv')
vecm_forecast_df[['btc_logp']].to_csv('results/tables/forecasts_vecm_btc.csv')
vecm_forecast_df[['eth_logp']].to_csv('results/tables/forecasts_vecm_eth.csv')

# Plots
fig, ax = plt.subplots(figsize=(12, 6))
test_var['btc_ret'].plot(ax=ax, label='Réel')
var_forecast_df['btc_ret'].plot(ax=ax, label='Prévision')
plt.title('Prévisions VAR: BTC Returns')
plt.legend()
plt.savefig('results/figures/03_previsions_var.png', dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
test_vecm['btc_logp'].plot(ax=ax, label='Réel')
vecm_forecast_df['btc_logp'].plot(ax=ax, label='Prévision')
plt.title('Prévisions VECM: BTC Log Price')
plt.legend()
plt.savefig('results/figures/03_previsions_vecm.png', dpi=150, bbox_inches='tight')
plt.close()

print('\n' + '='*60)
print('SECTION 9: IRF (Impulse Response Functions)')
print('='*60)

irf = var_res.irf(20)
fig = irf.plot()
plt.savefig('results/figures/03_irf.png', dpi=150, bbox_inches='tight')
plt.close()

print('\n' + '='*60)
print('SECTION 10: FEVD (Forecast Error Variance Decomposition)')
print('='*60)

fevd = var_res.fevd(20)
fig = fevd.plot()
plt.savefig('results/figures/03_fevd.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAnalyse VAR/VECM terminée avec succès !")
