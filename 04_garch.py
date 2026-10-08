import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model

warnings.filterwarnings('ignore')

# Création des dossiers
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

print('='*60)
print("1. DATA LOADING")
print('='*60)

# Chargement des données
df = pd.read_csv('data/btc_eth_processed.csv', parse_dates=['date'], index_col='date')

with open('data/split.json', 'r') as f:
    sp = json.load(f)

# Split train/test
train = df.loc[:sp['train_end']].copy()
test = df.loc[sp['test_start']:sp['test_end']].copy()

# On multiplie par 100 pour la stabilité numérique des modèles GARCH
train['btc_ret_100'] = train['btc_ret'] * 100
train['eth_ret_100'] = train['eth_ret'] * 100
test['btc_ret_100'] = test['btc_ret'] * 100
test['eth_ret_100'] = test['eth_ret'] * 100

train = train.dropna(subset=['btc_ret_100', 'eth_ret_100'])
test = test.dropna(subset=['btc_ret_100', 'eth_ret_100'])

btc_ret_train = train['btc_ret_100']
eth_ret_train = train['eth_ret_100']

print('='*60)
print("2. ARCH-LM TEST")
print('='*60)
# Test ARCH-LM de Engle sur les rendements bruts (justification du GARCH)
lm_btc, pval_btc, fval_btc, fpval_btc = het_arch(btc_ret_train, nlags=5)
lm_eth, pval_eth, fval_eth, fpval_eth = het_arch(eth_ret_train, nlags=5)

arch_results = pd.DataFrame({
    'Asset': ['BTC', 'ETH'],
    'LM_Stat': [lm_btc, lm_eth],
    'P_Value': [pval_btc, pval_eth]
})
arch_results.to_csv('results/tables/04_arch_lm.csv', index=False)
print("Test ARCH-LM :")
print(arch_results)

print('='*60)
print("3. GARCH(1,1) DISTRIBUTIONS (BTC)")
print('='*60)

dists = ['normal', 't', 'skewt']
res_btc_dists = []
models_btc = {}

for d in dists:
    am = arch_model(btc_ret_train, vol='Garch', p=1, q=1, dist=d)
    res = am.fit(disp='off')
    models_btc[d] = res
    res_btc_dists.append({'Dist': d, 'AIC': res.aic, 'BIC': res.bic, 'LogLik': res.loglikelihood})

df_btc_dists = pd.DataFrame(res_btc_dists).set_index('Dist')
df_btc_dists.to_csv('results/tables/04_garch_selection_btc.csv')
print(df_btc_dists)
best_dist_btc = df_btc_dists['AIC'].idxmin()
print(f"Meilleure distribution pour BTC (AIC): {best_dist_btc}")

print('='*60)
print("4. GARCH(1,1) DISTRIBUTIONS (ETH)")
print('='*60)

res_eth_dists = []
models_eth = {}

for d in dists:
    am = arch_model(eth_ret_train, vol='Garch', p=1, q=1, dist=d)
    res = am.fit(disp='off')
    models_eth[d] = res
    res_eth_dists.append({'Dist': d, 'AIC': res.aic, 'BIC': res.bic, 'LogLik': res.loglikelihood})

df_eth_dists = pd.DataFrame(res_eth_dists).set_index('Dist')
df_eth_dists.to_csv('results/tables/04_garch_selection_eth.csv')
print(df_eth_dists)
best_dist_eth = df_eth_dists['AIC'].idxmin()
print(f"Meilleure distribution pour ETH (AIC): {best_dist_eth}")


print('='*60)
print("5. HIGHER ORDER GARCH")
print('='*60)

orders = [(1,1), (1,2), (2,1), (2,2)]
order_res = []

for p, q in orders:
    # BTC
    am_btc = arch_model(btc_ret_train, vol='Garch', p=p, q=q, dist=best_dist_btc)
    res_b = am_btc.fit(disp='off')
    order_res.append({'Asset': 'BTC', 'Model': f'GARCH({p},{q})', 'AIC': res_b.aic, 'BIC': res_b.bic})
    
    # ETH
    am_eth = arch_model(eth_ret_train, vol='Garch', p=p, q=q, dist=best_dist_eth)
    res_e = am_eth.fit(disp='off')
    order_res.append({'Asset': 'ETH', 'Model': f'GARCH({p},{q})', 'AIC': res_e.aic, 'BIC': res_e.bic})

df_orders = pd.DataFrame(order_res)
df_orders.to_csv('results/tables/04_garch_ordres.csv', index=False)
print(df_orders)

# On garde le GARCH(1,1) ou le meilleur pour la suite. Simplification : on reste sur 1,1.
best_order_btc = (1,1)
best_order_eth = (1,1)


print('='*60)
print("6. ASYMMETRIC MODELS (EGARCH, GJR-GARCH)")
print('='*60)

asym_res = []

# BTC Asym
am_egarch_btc = arch_model(btc_ret_train, vol='EGARCH', p=1, o=1, q=1, dist=best_dist_btc)
res_egarch_btc = am_egarch_btc.fit(disp='off')

am_gjr_btc = arch_model(btc_ret_train, vol='Garch', p=1, o=1, q=1, dist=best_dist_btc)
res_gjr_btc = am_gjr_btc.fit(disp='off')

asym_res.append({'Asset': 'BTC', 'Model': 'GARCH(1,1)', 'AIC': models_btc[best_dist_btc].aic})
asym_res.append({'Asset': 'BTC', 'Model': 'EGARCH(1,1,1)', 'AIC': res_egarch_btc.aic})
asym_res.append({'Asset': 'BTC', 'Model': 'GJR-GARCH(1,1,1)', 'AIC': res_gjr_btc.aic})

# ETH Asym
am_egarch_eth = arch_model(eth_ret_train, vol='EGARCH', p=1, o=1, q=1, dist=best_dist_eth)
res_egarch_eth = am_egarch_eth.fit(disp='off')

am_gjr_eth = arch_model(eth_ret_train, vol='Garch', p=1, o=1, q=1, dist=best_dist_eth)
res_gjr_eth = am_gjr_eth.fit(disp='off')

asym_res.append({'Asset': 'ETH', 'Model': 'GARCH(1,1)', 'AIC': models_eth[best_dist_eth].aic})
asym_res.append({'Asset': 'ETH', 'Model': 'EGARCH(1,1,1)', 'AIC': res_egarch_eth.aic})
asym_res.append({'Asset': 'ETH', 'Model': 'GJR-GARCH(1,1,1)', 'AIC': res_gjr_eth.aic})

df_asym = pd.DataFrame(asym_res)
df_asym.to_csv('results/tables/04_garch_asymetrique.csv', index=False)
print(df_asym)


print('='*60)
print("7. DIAGNOSTICS OF STANDARDIZED RESIDUALS")
print('='*60)

best_model_btc = models_btc[best_dist_btc]
best_model_eth = models_eth[best_dist_eth]

std_resid_btc = best_model_btc.resid / best_model_btc.conditional_volatility
std_resid_eth = best_model_eth.resid / best_model_eth.conditional_volatility
std_resid_btc = std_resid_btc.dropna()
std_resid_eth = std_resid_eth.dropna()

# QQ-Plots
fig, ax = plt.subplots(figsize=(8,6))
sm.qqplot(std_resid_btc, line='s', ax=ax)
plt.title('QQ-Plot of Standardized Residuals (BTC)')
plt.savefig('results/figures/04_qq_residus_btc.png', dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(8,6))
sm.qqplot(std_resid_eth, line='s', ax=ax)
plt.title('QQ-Plot of Standardized Residuals (ETH)')
plt.savefig('results/figures/04_qq_residus_eth.png', dpi=150, bbox_inches='tight')
plt.close()

# ACF
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plot_acf(std_resid_btc, ax=axes[0,0], title='ACF Std Resid BTC')
plot_acf(std_resid_btc**2, ax=axes[0,1], title='ACF Squared Std Resid BTC')
plot_acf(std_resid_eth, ax=axes[1,0], title='ACF Std Resid ETH')
plot_acf(std_resid_eth**2, ax=axes[1,1], title='ACF Squared Std Resid ETH')
plt.tight_layout()
plt.savefig('results/figures/04_acf_residus_garch.png', dpi=150, bbox_inches='tight')
plt.close()

# Ljung-Box & ARCH-LM tests on residuals
diag_res = []

# Ljung Box
lb_btc = acorr_ljungbox(std_resid_btc, lags=[10], return_df=True)
lb_btc2 = acorr_ljungbox(std_resid_btc**2, lags=[10], return_df=True)
lb_eth = acorr_ljungbox(std_resid_eth, lags=[10], return_df=True)
lb_eth2 = acorr_ljungbox(std_resid_eth**2, lags=[10], return_df=True)

# ARCH-LM
arch_lm_btc_res = het_arch(std_resid_btc, nlags=5)
arch_lm_eth_res = het_arch(std_resid_eth, nlags=5)

diag_res.append({'Asset': 'BTC', 'Test': 'Ljung-Box (Resid)', 'Stat': lb_btc['lb_stat'].iloc[0], 'P-Value': lb_btc['lb_pvalue'].iloc[0]})
diag_res.append({'Asset': 'BTC', 'Test': 'Ljung-Box (Sq Resid)', 'Stat': lb_btc2['lb_stat'].iloc[0], 'P-Value': lb_btc2['lb_pvalue'].iloc[0]})
diag_res.append({'Asset': 'BTC', 'Test': 'ARCH-LM', 'Stat': arch_lm_btc_res[0], 'P-Value': arch_lm_btc_res[1]})
diag_res.append({'Asset': 'ETH', 'Test': 'Ljung-Box (Resid)', 'Stat': lb_eth['lb_stat'].iloc[0], 'P-Value': lb_eth['lb_pvalue'].iloc[0]})
diag_res.append({'Asset': 'ETH', 'Test': 'Ljung-Box (Sq Resid)', 'Stat': lb_eth2['lb_stat'].iloc[0], 'P-Value': lb_eth2['lb_pvalue'].iloc[0]})
diag_res.append({'Asset': 'ETH', 'Test': 'ARCH-LM', 'Stat': arch_lm_eth_res[0], 'P-Value': arch_lm_eth_res[1]})

df_diag = pd.DataFrame(diag_res)
df_diag.to_csv('results/tables/04_diagnostic_garch.csv', index=False)
print(df_diag)


print('='*60)
print("8. CONDITIONAL VOLATILITY VISUALIZATION")
print('='*60)

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
axes[0].plot(train.index, btc_ret_train, alpha=0.5, label='Returns', color='gray')
axes[0].plot(train.index, best_model_btc.conditional_volatility, label='Cond Vol', color='red')
axes[0].set_title('BTC Conditional Volatility vs Returns')
axes[0].legend()

axes[1].plot(train.index, eth_ret_train, alpha=0.5, label='Returns', color='gray')
axes[1].plot(train.index, best_model_eth.conditional_volatility, label='Cond Vol', color='blue')
axes[1].set_title('ETH Conditional Volatility vs Returns')
axes[1].legend()

plt.tight_layout()
plt.savefig('results/figures/04_volatilite_conditionnelle.png', dpi=150, bbox_inches='tight')
plt.close()

# Annulized Vol
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
axes[0].plot(train.index, best_model_btc.conditional_volatility * np.sqrt(365), color='red')
axes[0].set_title('BTC Annualized Volatility (%)')
axes[1].plot(train.index, best_model_eth.conditional_volatility * np.sqrt(365), color='blue')
axes[1].set_title('ETH Annualized Volatility (%)')
plt.tight_layout()
plt.savefig('results/figures/04_volatilite_annualisee.png', dpi=150, bbox_inches='tight')
plt.close()


print('='*60)
print("9. FORECASTING")
print('='*60)

all_dates = df.index
test_dates = test.index

# FIX : on retire NaN / inf (ex: premier rendement) avant de passer à arch_model
btc_full_ret = (df['btc_ret'] * 100).replace([np.inf, -np.inf], np.nan).dropna()
eth_full_ret = (df['eth_ret'] * 100).replace([np.inf, -np.inf], np.nan).dropna()

btc_var_forecast = []
eth_var_forecast = []

for i in range(len(test_dates)):
    current_date = test_dates[i]
    train_data_btc = btc_full_ret[:current_date].iloc[:-1] # Exclure la date courante
    train_data_eth = eth_full_ret[:current_date].iloc[:-1]
    
    # BTC
    am_b = arch_model(train_data_btc, vol='Garch', p=1, q=1, dist=best_dist_btc)
    res_b = am_b.fit(disp='off', update_freq=0, show_warning=False)
    f_b = res_b.forecast(horizon=1)
    btc_var_forecast.append(f_b.variance.iloc[-1, 0])
    
    # ETH
    am_e = arch_model(train_data_eth, vol='Garch', p=1, q=1, dist=best_dist_eth)
    res_e = am_e.fit(disp='off', update_freq=0, show_warning=False)
    f_e = res_e.forecast(horizon=1)
    eth_var_forecast.append(f_e.variance.iloc[-1, 0])

btc_var_forecast = np.array(btc_var_forecast)
eth_var_forecast = np.array(eth_var_forecast)

df_forecasts_btc = pd.DataFrame({'Date': test_dates, 'Forecast_Var': btc_var_forecast, 'Realized_Var': (test['btc_ret_100'])**2}).set_index('Date')
df_forecasts_eth = pd.DataFrame({'Date': test_dates, 'Forecast_Var': eth_var_forecast, 'Realized_Var': (test['eth_ret_100'])**2}).set_index('Date')

df_forecasts_btc.to_csv('results/tables/forecasts_garch_btc.csv')
df_forecasts_eth.to_csv('results/tables/forecasts_garch_eth.csv')

# RMSE
rmse_btc = np.sqrt(np.mean((df_forecasts_btc['Forecast_Var'] - df_forecasts_btc['Realized_Var'])**2))
rmse_eth = np.sqrt(np.mean((df_forecasts_eth['Forecast_Var'] - df_forecasts_eth['Realized_Var'])**2))

metrics_df = pd.DataFrame({
    'Asset': ['BTC', 'ETH'],
    'RMSE_Vol': [rmse_btc, rmse_eth]
})
metrics_df.to_csv('results/tables/04_metriques_garch.csv', index=False)
print("RMSE des prévisions de variance :")
print(metrics_df)

# Plot previsions vs real
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
axes[0].plot(df_forecasts_btc.index, np.sqrt(df_forecasts_btc['Realized_Var']), alpha=0.5, label='Realized Vol')
axes[0].plot(df_forecasts_btc.index, np.sqrt(df_forecasts_btc['Forecast_Var']), label='Forecast Vol', color='red')
axes[0].set_title('BTC: Forecast vs Realized Volatility')
axes[0].legend()

axes[1].plot(df_forecasts_eth.index, np.sqrt(df_forecasts_eth['Realized_Var']), alpha=0.5, label='Realized Vol')
axes[1].plot(df_forecasts_eth.index, np.sqrt(df_forecasts_eth['Forecast_Var']), label='Forecast Vol', color='blue')
axes[1].set_title('ETH: Forecast vs Realized Volatility')
axes[1].legend()

plt.tight_layout()
plt.savefig('results/figures/04_previsions_garch.png', dpi=150, bbox_inches='tight')
plt.close()

print('='*60)
print("10. VALUE AT RISK (VaR)")
print('='*60)

am_btc_final = arch_model(btc_full_ret[:sp['test_end']], vol='Garch', p=1, q=1, dist=best_dist_btc).fit(disp='off')
am_eth_final = arch_model(eth_full_ret[:sp['test_end']], vol='Garch', p=1, q=1, dist=best_dist_eth).fit(disp='off')

# VaR via quantile de la distribution normale (approximation)
# VaR = mu + sigma * z_alpha
z_05 = stats.norm.ppf(0.05)
z_01 = stats.norm.ppf(0.01)

var_5_btc = np.sqrt(df_forecasts_btc['Forecast_Var']) * z_05
var_1_btc = np.sqrt(df_forecasts_btc['Forecast_Var']) * z_01

var_5_eth = np.sqrt(df_forecasts_eth['Forecast_Var']) * z_05
var_1_eth = np.sqrt(df_forecasts_eth['Forecast_Var']) * z_01

df_forecasts_btc['VaR_5'] = var_5_btc
df_forecasts_btc['VaR_1'] = var_1_btc
df_forecasts_eth['VaR_5'] = var_5_eth
df_forecasts_eth['VaR_1'] = var_1_eth

# VaR Violations
btc_ret_test = test['btc_ret_100']
eth_ret_test = test['eth_ret_100']

viol_5_btc = (btc_ret_test < df_forecasts_btc['VaR_5']).mean()
viol_1_btc = (btc_ret_test < df_forecasts_btc['VaR_1']).mean()
viol_5_eth = (eth_ret_test < df_forecasts_eth['VaR_5']).mean()
viol_1_eth = (eth_ret_test < df_forecasts_eth['VaR_1']).mean()

var_viols = pd.DataFrame({
    'Asset': ['BTC', 'BTC', 'ETH', 'ETH'],
    'VaR_Level': ['5%', '1%', '5%', '1%'],
    'Violation_Rate': [viol_5_btc, viol_1_btc, viol_5_eth, viol_1_eth],
    'Expected_Rate': [0.05, 0.01, 0.05, 0.01]
})
var_viols.to_csv('results/tables/04_var_violations.csv', index=False)
print("Violations VaR :")
print(var_viols)

# Plot VaR
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
axes[0].plot(df_forecasts_btc.index, btc_ret_test, label='Returns', alpha=0.6, color='gray')
axes[0].plot(df_forecasts_btc.index, df_forecasts_btc['VaR_5'], label='VaR 5%', color='orange', linestyle='--')
axes[0].plot(df_forecasts_btc.index, df_forecasts_btc['VaR_1'], label='VaR 1%', color='red', linestyle='--')
axes[0].set_title('BTC Returns vs Value at Risk')
axes[0].legend()

axes[1].plot(df_forecasts_eth.index, eth_ret_test, label='Returns', alpha=0.6, color='gray')
axes[1].plot(df_forecasts_eth.index, df_forecasts_eth['VaR_5'], label='VaR 5%', color='orange', linestyle='--')
axes[1].plot(df_forecasts_eth.index, df_forecasts_eth['VaR_1'], label='VaR 1%', color='red', linestyle='--')
axes[1].set_title('ETH Returns vs Value at Risk')
axes[1].legend()

plt.tight_layout()
plt.savefig('results/figures/04_var_risk.png', dpi=150, bbox_inches='tight')
plt.close()

print('='*60)
print("ANALYSE GARCH TERMINEE")
print('='*60)