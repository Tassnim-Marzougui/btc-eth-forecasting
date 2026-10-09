"""
04_garch.py : Personne 2
ARCH-LM, GARCH(1,1) (normal / t / skew-t), ordres superieurs, EGARCH / GJR,
diagnostic des residus standardises, volatilite conditionnelle,
previsions de VARIANCE evaluees A PART (pas de conversion en prix),
Value-at-Risk avec test de Kupiec.

Sorties principales :
  results/tables/forecasts_P2_garch.csv   (walk-forward sur les origines communes, h=1..H, en variance)
  results/tables/04_metriques_garch.csv   (RMSE / MAE / QLIKE, GARCH vs EWMA vs variance historique)
  results/tables/04_var_violations.csv    (taux de violations + test de Kupiec)
"""
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

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

# ----------------------------------------------------------------------
# Parametres (a aligner avec 03_var_coint.py / P1 / P3)
# ----------------------------------------------------------------------
H = 10                       # horizon multi-pas (variance)
N_ORIGINS = 32               # nombre d'origines walk-forward
ORIGINS_FILE = 'data/origins.json'   # meme liste que dans 03_var_coint.py si fournie
REFIT_EVERY = 1              # re-estimation tous les k jours pour le VaR 1 pas (1 = quotidien)
EWMA_LAMBDA = 0.94           # RiskMetrics
HIST_WINDOW = 30             # fenetre de la variance historique (jours)

# ======================================================================
print('=' * 60)
print("1. DATA LOADING")
print('=' * 60)
# ======================================================================
df = pd.read_csv('data/btc_eth_processed.csv', parse_dates=['date'], index_col='date')

with open('data/split.json', 'r') as f:
    sp = json.load(f)

train = df.loc[:sp['train_end']].copy()
test = df.loc[sp['test_start']:sp['test_end']].copy()

# x100 pour la stabilite numerique des modeles GARCH
for a in ['btc', 'eth']:
    train[f'{a}_ret_100'] = train[f'{a}_ret'] * 100
    test[f'{a}_ret_100'] = test[f'{a}_ret'] * 100

train = train.dropna(subset=['btc_ret_100', 'eth_ret_100'])
test = test.dropna(subset=['btc_ret_100', 'eth_ret_100'])

btc_ret_train = train['btc_ret_100']
eth_ret_train = train['eth_ret_100']

# Series de rendements complets (pour le walk-forward)
ret_full = {
    'BTC': (df['btc_ret'] * 100).replace([np.inf, -np.inf], np.nan).dropna(),
    'ETH': (df['eth_ret'] * 100).replace([np.inf, -np.inf], np.nan).dropna(),
}

# ======================================================================
print('\n' + '=' * 60)
print("2. ARCH-LM TEST")
print('=' * 60)
# ======================================================================
lm_btc, pval_btc, _, _ = het_arch(btc_ret_train, nlags=5)
lm_eth, pval_eth, _, _ = het_arch(eth_ret_train, nlags=5)

arch_results = pd.DataFrame({
    'Asset': ['BTC', 'ETH'],
    'LM_Stat': [lm_btc, lm_eth],
    'P_Value': [pval_btc, pval_eth],
})
arch_results.to_csv('results/tables/04_arch_lm.csv', index=False)
print(arch_results)

# ======================================================================
print('\n' + '=' * 60)
print("3-4. GARCH(1,1) : CHOIX DE LA DISTRIBUTION")
print('=' * 60)
# ======================================================================
dists = ['normal', 't', 'skewt']

def select_distribution(returns, label):
    rows, models = [], {}
    for d in dists:
        res = arch_model(returns, vol='Garch', p=1, q=1, dist=d).fit(disp='off')
        models[d] = res
        rows.append({'Dist': d, 'AIC': res.aic, 'BIC': res.bic, 'LogLik': res.loglikelihood})
    out = pd.DataFrame(rows).set_index('Dist')
    out.to_csv(f'results/tables/04_garch_selection_{label.lower()}.csv')
    print(f"\n{label}:")
    print(out)
    best = out['AIC'].idxmin()
    print(f"Meilleure distribution {label} (AIC): {best}")
    return models, best

models_btc, best_dist_btc = select_distribution(btc_ret_train, 'BTC')
models_eth, best_dist_eth = select_distribution(eth_ret_train, 'ETH')

best_dist = {'BTC': best_dist_btc, 'ETH': best_dist_eth}

# ======================================================================
print('\n' + '=' * 60)
print("5. ORDRES SUPERIEURS")
print('=' * 60)
# ======================================================================
orders = [(1, 1), (1, 2), (2, 1), (2, 2)]
order_rows = []
for p, q in orders:
    for asset, series in [('BTC', btc_ret_train), ('ETH', eth_ret_train)]:
        res = arch_model(series, vol='Garch', p=p, q=q, dist=best_dist[asset]).fit(disp='off')
        order_rows.append({'Asset': asset, 'Model': f'GARCH({p},{q})', 'p': p, 'q': q,
                           'AIC': res.aic, 'BIC': res.bic})

df_orders = pd.DataFrame(order_rows)
df_orders.to_csv('results/tables/04_garch_ordres.csv', index=False)
print(df_orders[['Asset', 'Model', 'AIC', 'BIC']])

# Regle : on ne quitte (1,1) que si un ordre superieur ameliore le BIC d'au moins 2 points
best_order = {}
for asset in ['BTC', 'ETH']:
    sub = df_orders[df_orders.Asset == asset].set_index(['p', 'q'])
    bic_11 = sub.loc[(1, 1), 'BIC']
    cand = sub['BIC'].idxmin()
    best_order[asset] = cand if (bic_11 - sub.loc[cand, 'BIC']) >= 2 else (1, 1)
    print(f"Ordre retenu {asset}: GARCH{best_order[asset]}")

# ======================================================================
print('\n' + '=' * 60)
print("6. MODELES ASYMETRIQUES (EGARCH, GJR-GARCH)")
print('=' * 60)
# ======================================================================
asym_rows = []
asym_params = []
for asset, series, models in [('BTC', btc_ret_train, models_btc), ('ETH', eth_ret_train, models_eth)]:
    d = best_dist[asset]
    res_e = arch_model(series, vol='EGARCH', p=1, o=1, q=1, dist=d).fit(disp='off')
    res_g = arch_model(series, vol='Garch', p=1, o=1, q=1, dist=d).fit(disp='off')
    base = models[d]
    asym_rows += [
        {'Asset': asset, 'Model': 'GARCH(1,1)', 'AIC': base.aic, 'BIC': base.bic},
        {'Asset': asset, 'Model': 'EGARCH(1,1,1)', 'AIC': res_e.aic, 'BIC': res_e.bic},
        {'Asset': asset, 'Model': 'GJR-GARCH(1,1,1)', 'AIC': res_g.aic, 'BIC': res_g.bic},
    ]
    # parametre de levier (gamma[1]) : significatif => effet asymetrique
    asym_params.append({'Asset': asset, 'Model': 'EGARCH', 'gamma': res_e.params.get('gamma[1]', np.nan),
                        'p_value': res_e.pvalues.get('gamma[1]', np.nan)})
    asym_params.append({'Asset': asset, 'Model': 'GJR', 'gamma': res_g.params.get('gamma[1]', np.nan),
                        'p_value': res_g.pvalues.get('gamma[1]', np.nan)})

df_asym = pd.DataFrame(asym_rows)
df_asym.to_csv('results/tables/04_garch_asymetrique.csv', index=False)
pd.DataFrame(asym_params).to_csv('results/tables/04_garch_levier.csv', index=False)
print(df_asym)
print(pd.DataFrame(asym_params))

# ======================================================================
print('\n' + '=' * 60)
print("7. DIAGNOSTIC DES RESIDUS STANDARDISES")
print('=' * 60)
# ======================================================================
# Modele diagnostique : ordre retenu + meilleure distribution
final_models = {}
for asset, series in [('BTC', btc_ret_train), ('ETH', eth_ret_train)]:
    p, q = best_order[asset]
    final_models[asset] = arch_model(series, vol='Garch', p=p, q=q, dist=best_dist[asset]).fit(disp='off')

std_resid = {a: (final_models[a].resid / final_models[a].conditional_volatility).dropna()
             for a in ['BTC', 'ETH']}

for a in ['BTC', 'ETH']:
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.qqplot(std_resid[a], line='s', ax=ax)
    plt.title(f'QQ-Plot of Standardized Residuals ({a})')
    plt.savefig(f'results/figures/04_qq_residus_{a.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plot_acf(std_resid['BTC'], ax=axes[0, 0], title='ACF Std Resid BTC')
plot_acf(std_resid['BTC'] ** 2, ax=axes[0, 1], title='ACF Squared Std Resid BTC')
plot_acf(std_resid['ETH'], ax=axes[1, 0], title='ACF Std Resid ETH')
plot_acf(std_resid['ETH'] ** 2, ax=axes[1, 1], title='ACF Squared Std Resid ETH')
plt.tight_layout()
plt.savefig('results/figures/04_acf_residus_garch.png', dpi=150, bbox_inches='tight')
plt.close()

diag_rows = []
for a in ['BTC', 'ETH']:
    z = std_resid[a]
    lb = acorr_ljungbox(z, lags=[10], return_df=True)
    lb2 = acorr_ljungbox(z ** 2, lags=[10], return_df=True)
    lm = het_arch(z, nlags=5)
    jb_stat, jb_p = stats.jarque_bera(z)
    diag_rows += [
        {'Asset': a, 'Test': 'Ljung-Box (Resid)', 'Stat': lb['lb_stat'].iloc[0], 'P-Value': lb['lb_pvalue'].iloc[0]},
        {'Asset': a, 'Test': 'Ljung-Box (Sq Resid)', 'Stat': lb2['lb_stat'].iloc[0], 'P-Value': lb2['lb_pvalue'].iloc[0]},
        {'Asset': a, 'Test': 'ARCH-LM', 'Stat': lm[0], 'P-Value': lm[1]},
        {'Asset': a, 'Test': 'Jarque-Bera', 'Stat': jb_stat, 'P-Value': jb_p},
    ]
df_diag = pd.DataFrame(diag_rows)
df_diag.to_csv('results/tables/04_diagnostic_garch.csv', index=False)
print(df_diag)

# ======================================================================
print('\n' + '=' * 60)
print("8. VOLATILITE CONDITIONNELLE")
print('=' * 60)
# ======================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
for ax, a, series, col in [(axes[0], 'BTC', btc_ret_train, 'red'), (axes[1], 'ETH', eth_ret_train, 'blue')]:
    ax.plot(train.index, series, alpha=0.5, label='Returns (%)', color='gray')
    ax.plot(train.index, final_models[a].conditional_volatility, label='Cond Vol', color=col)
    ax.set_title(f'{a} Conditional Volatility vs Returns')
    ax.legend()
plt.tight_layout()
plt.savefig('results/figures/04_volatilite_conditionnelle.png', dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
for ax, a, col in [(axes[0], 'BTC', 'red'), (axes[1], 'ETH', 'blue')]:
    ax.plot(train.index, final_models[a].conditional_volatility * np.sqrt(365), color=col)
    ax.set_title(f'{a} Annualized Volatility (%)')
plt.tight_layout()
plt.savefig('results/figures/04_volatilite_annualisee.png', dpi=150, bbox_inches='tight')
plt.close()

# ======================================================================
# Outils de prevision et d'evaluation (variance)
# ======================================================================
def garch_spec(series, asset):
    p, q = best_order[asset]
    return arch_model(series, vol='Garch', p=p, q=q, dist=best_dist[asset])

def qlike(realized_proxy, sigma2):
    """Perte QLIKE (a minimiser) : log(sigma2) + r2/sigma2 (constante omise)."""
    sigma2 = np.maximum(sigma2, 1e-12)
    return np.log(sigma2) + realized_proxy / sigma2

def vol_metrics(g):
    e = g['y_true'] - g['y_pred']
    return pd.Series({
        'RMSE_var': np.sqrt((e ** 2).mean()),
        'MAE_var': e.abs().mean(),
        'QLIKE': qlike(g['y_true'].values, g['y_pred'].values).mean(),
    })

def ewma_variance(returns, lam=EWMA_LAMBDA):
    """Variance EWMA (RiskMetrics) : dernier sigma2 connu, prevision plate sur l'horizon."""
    r = returns.values
    s2 = np.var(r[:30]) if len(r) >= 30 else np.var(r)
    for x in r:
        s2 = lam * s2 + (1 - lam) * x ** 2
    return s2

# ======================================================================
print('\n' + '=' * 60)
print("9. PREVISIONS DE VARIANCE : WALK-FORWARD (origines communes, h=1..%d)" % H)
print('=' * 60)
# ======================================================================
# Meme logique d'origines que 03_var_coint.py : t = nb d'observations dans le train
full = df[['btc_logp', 'eth_logp']].dropna()

if os.path.exists(ORIGINS_FILE):
    with open(ORIGINS_FILE, 'r') as f:
        origins = np.array(json.load(f), dtype=int)
    print(f"Origines chargees depuis {ORIGINS_FILE}: {len(origins)}")
else:
    test_start_pos = full.index.get_loc(pd.Timestamp(sp['test_start']))
    last_origin = len(full) - H
    origins = np.linspace(test_start_pos, last_origin, N_ORIGINS).astype(int)
    print(f"Origines generees ({len(origins)}) : A REMPLACER par la liste commune (data/origins.json).")

rows = []
for t in origins:
    info_end = full.index[t - 1]                 # derniere date connue a l'origine
    fut_dates = full.index[t:t + H]
    if len(fut_dates) < H:
        continue
    for asset in ['BTC', 'ETH']:
        r_all = ret_full[asset]
        r_train = r_all[r_all.index <= info_end]
        r_true = r_all.reindex(fut_dates)
        if r_true.isna().any():
            continue

        # GARCH : variance previsionnelle h=1..H
        res = garch_spec(r_train, asset).fit(disp='off', show_warning=False)
        fvar = res.forecast(horizon=H, reindex=False).variance.values[-1]   # (H,)

        # References : EWMA et variance historique (previsions plates)
        s2_ewma = ewma_variance(r_train)
        s2_hist = np.var(r_train.iloc[-HIST_WINDOW:])

        for h in range(H):
            y = r_true.iloc[h] ** 2
            for name, val in [('GARCH', fvar[h]), ('EWMA', s2_ewma), ('HIST', s2_hist)]:
                rows.append((int(t), fut_dates[h], h + 1, asset, name, y, val))

fc = pd.DataFrame(rows, columns=['origin', 'date', 'h', 'asset', 'model', 'y_true', 'y_pred'])
fc.to_csv('results/tables/forecasts_P2_garch.csv', index=False)
print(f"Export : results/tables/forecasts_P2_garch.csv ({len(fc)} lignes)")
print("NB : y_true = rendement^2 (%^2) : proxy de la variance realisee ; y_pred = variance prevue.")

metrics_df = fc.groupby(['model', 'asset']).apply(vol_metrics).reset_index()
metrics_df.to_csv('results/tables/04_metriques_garch.csv', index=False)
print("\nMetriques de variance (walk-forward, h=1..%d):" % H)
print(metrics_df)

by_h = fc.groupby(['model', 'asset', 'h']).apply(vol_metrics).reset_index()
by_h.to_csv('results/tables/04_metriques_garch_par_horizon.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, a in zip(axes, ['BTC', 'ETH']):
    for m in ['GARCH', 'EWMA', 'HIST']:
        s = by_h[(by_h.model == m) & (by_h.asset == a)]
        ax.plot(s['h'], s['QLIKE'], marker='o', label=m)
    ax.set_title(f'QLIKE par horizon ({a})')
    ax.set_xlabel('Horizon h')
    ax.legend()
plt.tight_layout()
plt.savefig('results/figures/04_qlike_par_horizon.png', dpi=150, bbox_inches='tight')
plt.close()

# ======================================================================
print('\n' + '=' * 60)
print("10. PREVISION 1 PAS QUOTIDIENNE SUR TOUT LE TEST + VALUE AT RISK")
print('=' * 60)
# ======================================================================
def std_quantiles(res, levels=(0.05, 0.01)):
    """Quantiles de la distribution standardisee estimee (normal / t / skew-t)."""
    dist = res.model.distribution
    k = dist.num_params
    dparams = res.params.iloc[-k:].values if k > 0 else np.array([])
    return dist.ppf(np.array(levels), dparams)

daily = {}
for asset in ['BTC', 'ETH']:
    r_all = ret_full[asset]
    recs, params_prev = [], None
    for i, d in enumerate(test.index):
        r_train = r_all[r_all.index < d]            # information strictement anterieure a d
        am = garch_spec(r_train, asset)
        if params_prev is None or i % REFIT_EVERY == 0:
            res = am.fit(disp='off', update_freq=0, show_warning=False)
            params_prev = res.params.values
        else:
            res = am.fix(params_prev)               # parametres figes, filtre mis a jour
        f = res.forecast(horizon=1, reindex=False)
        sigma2 = f.variance.values[-1, 0]
        mu = f.mean.values[-1, 0]
        q05, q01 = std_quantiles(res)
        sig = np.sqrt(sigma2)
        recs.append({
            'Date': d,
            'Forecast_Var': sigma2,
            'Realized_Var': r_all.loc[d] ** 2,
            'Return': r_all.loc[d],
            'EWMA_Var': ewma_variance(r_train),
            'HIST_Var': np.var(r_train.iloc[-HIST_WINDOW:]),
            'VaR_5': mu + sig * q05,
            'VaR_1': mu + sig * q01,
        })
    out = pd.DataFrame(recs).set_index('Date')
    out.to_csv(f'results/tables/forecasts_garch_{asset.lower()}.csv')
    daily[asset] = out

# Metriques 1 pas (GARCH vs benchmarks)
m_rows = []
for asset in ['BTC', 'ETH']:
    o = daily[asset]
    for name, col in [('GARCH', 'Forecast_Var'), ('EWMA', 'EWMA_Var'), ('HIST', 'HIST_Var')]:
        e = o['Realized_Var'] - o[col]
        m_rows.append({'model': name, 'asset': asset,
                       'RMSE_var': np.sqrt((e ** 2).mean()), 'MAE_var': e.abs().mean(),
                       'QLIKE': qlike(o['Realized_Var'].values, o[col].values).mean()})
df_m1 = pd.DataFrame(m_rows)
df_m1.to_csv('results/tables/04_metriques_garch_1pas.csv', index=False)
print("Metriques de variance a 1 pas (tout le test):")
print(df_m1)

# Graphique : volatilite prevue vs proxy realise
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
for ax, a, col in [(axes[0], 'BTC', 'red'), (axes[1], 'ETH', 'blue')]:
    o = daily[a]
    ax.plot(o.index, np.sqrt(o['Realized_Var']), alpha=0.4, label='|rendement| (proxy realise)')
    ax.plot(o.index, np.sqrt(o['Forecast_Var']), color=col, label='Vol prevue GARCH')
    ax.plot(o.index, np.sqrt(o['EWMA_Var']), color='green', alpha=0.7, label='Vol EWMA')
    ax.set_title(f'{a}: volatilite prevue vs realisee (%)')
    ax.legend()
plt.tight_layout()
plt.savefig('results/figures/04_previsions_garch.png', dpi=150, bbox_inches='tight')
plt.close()

# ---- Value at Risk : quantiles de la distribution estimee + test de Kupiec
def kupiec_pof(n_viol, n, p):
    """Test de couverture inconditionnelle de Kupiec (LR ~ chi2(1))."""
    pi = n_viol / n
    ll0 = (n - n_viol) * np.log(1 - p) + n_viol * np.log(p)
    ll1 = 0.0
    if n_viol < n:
        ll1 += (n - n_viol) * np.log(1 - pi)
    if n_viol > 0:
        ll1 += n_viol * np.log(pi)
    lr = -2 * (ll0 - ll1)
    return lr, 1 - stats.chi2.cdf(lr, 1)

viol_rows = []
for asset in ['BTC', 'ETH']:
    o = daily[asset]
    for lvl, col, p in [('5%', 'VaR_5', 0.05), ('1%', 'VaR_1', 0.01)]:
        v = int((o['Return'] < o[col]).sum())
        n = len(o)
        lr, pv = kupiec_pof(v, n, p)
        viol_rows.append({'Asset': asset, 'VaR_Level': lvl, 'N': n, 'Violations': v,
                          'Violation_Rate': v / n, 'Expected_Rate': p,
                          'Kupiec_LR': lr, 'Kupiec_p_value': pv})
var_viols = pd.DataFrame(viol_rows)
var_viols.to_csv('results/tables/04_var_violations.csv', index=False)
print("\nViolations VaR (quantiles de la distribution estimee) :")
print(var_viols)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
for ax, a in [(axes[0], 'BTC'), (axes[1], 'ETH')]:
    o = daily[a]
    ax.plot(o.index, o['Return'], label='Returns (%)', alpha=0.6, color='gray')
    ax.plot(o.index, o['VaR_5'], label='VaR 5%', color='orange', linestyle='--')
    ax.plot(o.index, o['VaR_1'], label='VaR 1%', color='red', linestyle='--')
    ax.set_title(f'{a} Returns vs Value at Risk')
    ax.legend()
plt.tight_layout()
plt.savefig('results/figures/04_var_risk.png', dpi=150, bbox_inches='tight')
plt.close()

print('\n' + '=' * 60)
print("ANALYSE GARCH TERMINEE")
print('=' * 60)