"""07 — Tableau final de comparaison + tests de Diebold-Mariano (tous les modèles, prix).

À lancer à la racine du dépôt :  python 07_tableau_final.py
Lit  : results/tables/forecasts_*_{btc,eth}.csv  et  results/tables/forecasts_P2.csv (VAR, VECM)
Écrit: results/tables/07_tableau_final.csv, 07_metriques_par_horizon.csv,
       07_dm_matrice_pvalues.csv, 07_dm_vs_naive.csv, 07_volatilite_garch.csv
Même protocole pour tous : 32 origines (rolling), horizon 10 jours, erreurs sur les PRIX.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

T = Path("results/tables")
H = 10
ASSETS = ["btc", "eth"]
STAT = ["naive", "drift", "arima_aic", "arima_bic"]
DL = ["lstm", "gru"]


def load_all(a):
    d = {n: pd.read_csv(T / f"forecasts_{n}_{a}.csv")[["origin", "h", "y_true", "y_pred"]] for n in STAT + DL}
    # P2 : origine entière -> date, même clé que les autres fichiers
    dates = pd.read_csv(next(Path(".").rglob("btc_eth.csv")), parse_dates=["date"])["date"]
    p2 = pd.read_csv(T / "forecasts_P2.csv")
    p2["origin"] = p2["origin"].map(lambda i: dates[i].strftime("%Y-%m-%d"))
    for m in ["VAR", "VECM"]:
        d[m.lower()] = p2[(p2.asset == a.upper()) & (p2.model == m)][["origin", "h", "y_true", "y_pred"]]
    return {k: v.set_index(["origin", "h"]).sort_index() for k, v in d.items()}


def dm_test(e1, e2, lag):
    """Diebold-Mariano, perte quadratique, correction de Harvey. Stat > 0 : modèle 1 moins bon."""
    d = e1 ** 2 - e2 ** 2
    n, db = len(d), d.mean()
    dc = d - db
    v = (dc ** 2).mean()
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (dc[k:] * dc[:-k]).mean()
    if v <= 0 or np.allclose(d, 0):
        return np.nan, np.nan
    hh = lag + 1
    s = db / np.sqrt(v / n) * np.sqrt((n + 1 - 2 * hh + hh * (hh - 1) / n) / n)
    return s, 2 * (1 - stats.t.cdf(abs(s), df=n - 1))


rows, hrows, mats, vs_naive = [], [], [], []
for a in ASSETS:
    d = load_all(a)
    err = {k: v["y_true"] - v["y_pred"] for k, v in d.items()}
    for k, v in d.items():
        e = err[k]
        rows.append({"actif": a.upper(), "modele": k, "MSE": (e ** 2).mean(), "RMSE": np.sqrt((e ** 2).mean()), "MAE": e.abs().mean(),
                     "MAPE_%": (e.abs() / v["y_true"]).mean() * 100, "n": len(e)})
        for h in (1, 5, 10):
            eh = e.xs(h, level="h")
            hrows.append({"actif": a.upper(), "modele": k, "h": h, "MSE": (eh ** 2).mean(), "RMSE": np.sqrt((eh ** 2).mean()),
                          "MAE": eh.abs().mean(), "MAPE_%": (eh.abs() / d[k].xs(h, level="h")["y_true"]).mean() * 100})
    names = list(d)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for m1 in names:
        for m2 in names:
            if m1 == m2:
                continue
            idx = err[m1].index.intersection(err[m2].index)
            M.loc[m1, m2] = dm_test(err[m1].loc[idx].values, err[m2].loc[idx].values, H - 1)[1]
    M.insert(0, "actif", a.upper())
    mats.append(M.rename_axis("modele1").reset_index())
    for m in names:
        if m == "naive":
            continue
        idx = err[m].index.intersection(err["naive"].index)
        s, p = dm_test(err[m].loc[idx].values, err["naive"].loc[idx].values, H - 1)
        r = {"actif": a.upper(), "modele": m, "h": "tous", "DM_stat": s, "p_value": p}
        vs_naive.append(r)
        for h in (1, 5, 10):
            s, p = dm_test(err[m].xs(h, level="h").values, err["naive"].xs(h, level="h").values, 0)
            vs_naive.append({"actif": a.upper(), "modele": m, "h": h, "DM_stat": s, "p_value": p})

final = pd.DataFrame(rows)
final["rang_RMSE"] = final.groupby("actif")["RMSE"].rank().astype(int)
final = final.sort_values(["actif", "RMSE"]).round(4)
final.to_csv(T / "07_tableau_final.csv", index=False)
pd.DataFrame(hrows).round(4).to_csv(T / "07_metriques_par_horizon.csv", index=False)
pd.concat(mats).round(4).to_csv(T / "07_dm_matrice_pvalues.csv", index=False)
dmn = pd.DataFrame(vs_naive).round(4)
dmn["conclusion_5%"] = np.where(dmn["p_value"].isna(), "identique à Naive (mêmes prévisions)",
                       np.where(dmn["p_value"] >= 0.05, "pas de différence significative",
                                np.where(dmn["DM_stat"] > 0, "moins bon que Naive", "meilleur que Naive")))
dmn.to_csv(T / "07_dm_vs_naive.csv", index=False)
if (T / "04_metriques_garch.csv").exists():  # GARCH : volatilité, pas de prix -> bloc séparé
    pd.read_csv(T / "04_metriques_garch.csv").round(4).to_csv(T / "07_volatilite_garch.csv", index=False)

print(final.to_string(index=False))
print()
print(dmn[dmn.h == "tous"].to_string(index=False))