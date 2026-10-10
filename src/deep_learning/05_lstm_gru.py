# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 05 — LSTM / GRU : prévision BTC & ETH (Personne 3)
# #
# Protocole identique à P1/P2 :
# - même split chronologique (`data/split.json`) : train jusqu'au 2025-11-14, test ensuite
# - rolling origin : une origine tous les 10 jours dans le test, horizon H = 10 pas (32 origines)
# - prévisions exportées au même format CSV que P1 : origin, date, h, y_true, y_pred, lo95, hi95
# #
# Choix méthodologiques (à justifier dans le rapport) :
# - entrée = rendements logarithmiques standardisés (moyenne/écart-type calculés sur le TRAIN seulement
#   -> pas de fuite d'information). Les prix bruts sortent de la plage d'apprentissage dans le test,
#   un LSTM/GRU extrapole très mal dans ce cas.
# - sortie multi-horizon directe : le réseau prédit les 10 prochains rendements d'un coup,
#   puis on reconstruit le prix : P_origine-1 * exp(cumsum(rendements prédits)).
# - hyperparamètres choisis par Time Series Cross-Validation (TimeSeriesSplit) sur le TRAIN uniquement.
# - évaluation finale par walk-forward (fenêtre expansive, fine-tuning à chaque origine).
# #
# Kaggle : Settings > Accelerator > GPU (T4 ou P100), puis "Run All".

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-10-10T11:23:29.947204Z","iopub.execute_input":"2026-10-10T11:23:29.947615Z","iopub.status.idle":"2026-10-10T11:23:39.873169Z","shell.execute_reply.started":"2026-10-10T11:23:29.947571Z","shell.execute_reply":"2026-10-10T11:23:39.872287Z"}}
import json
import random
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------------------- paramètres ---------------------------------
QUICK = False            # True = test rapide (petite grille, peu d'époques, 4 origines)
SEED = 42
H = 10                   # horizon de prévision (comme P1)
STEP_ORIGIN = 10         # une origine tous les 10 jours (comme P1)
BATCH = 64
LR = 1e-3
EPOCHS_FULL = 80         # entraînement initial
PATIENCE_FULL = 10
EPOCHS_FT = 15           # fine-tuning à chaque nouvelle origine
PATIENCE_FT = 3
VAL_FRAC = 0.15
USE_OTHER_ASSET = False  # True = ajoute les rendements de l'autre crypto en entrée (multivarié)
RUN_CV = True            # False = utilise DEFAULT_CFG sans chercher
DEFAULT_CFG = dict(lookback=60, units=64, layers=1, dropout=0.2)
GRID = dict(lookback=[30, 60], units=[32, 64], layers=[1, 2], dropout=[0.1, 0.3])
CV_SPLITS = 3
ASSETS = ["btc", "eth"]
MODELS = ["LSTM", "GRU"]

if QUICK:
    GRID = dict(lookback=[30], units=[32], layers=[1], dropout=[0.1])
    CV_SPLITS, EPOCHS_FULL, EPOCHS_FT = 2, 5, 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", DEVICE)


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_seed()

# ----------------------------- chemins ------------------------------------
def find(name):
    for root in [Path("/kaggle/input"), Path("."), Path("..")]:
        if root.exists():
            hits = sorted(root.rglob(name))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"{name} introuvable : ajoute le dépôt comme Dataset Kaggle.")


OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
FIG = OUT / "results" / "figures"
TAB = OUT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Données et split (identiques à P1/P2)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-10-10T11:23:39.874352Z","iopub.execute_input":"2026-10-10T11:23:39.874982Z","iopub.status.idle":"2026-10-10T11:23:40.056986Z","shell.execute_reply.started":"2026-10-10T11:23:39.874951Z","shell.execute_reply":"2026-10-10T11:23:40.056332Z"}}
df = pd.read_csv(find("btc_eth.csv"), parse_dates=["date"]).sort_values("date").set_index("date")
split = json.load(open(find("split.json")))
dates = df.index
n_train = int(split["n_train"])
assert dates[n_train] == pd.Timestamp(split["test_start"]), "split.json incohérent avec les données"
n = len(df)
origin_pos = [p for p in range(n_train, n, STEP_ORIGIN) if p + H <= n]
if QUICK:
    origin_pos = origin_pos[:4]
print(f"{n} jours | train={n_train} | test={n - n_train} | {len(origin_pos)} origines, horizon {H}")

# rendements log alignés : r[k] = log P[k+1] - log P[k]
data = {}
for a in ASSETS:
    P = df[f"{a}_close"].values.astype(float)
    r = np.diff(np.log(P))
    data[a] = dict(P=P, r=r, mu=r[: n_train - 1].mean(), sd=r[: n_train - 1].std())
    print(a.upper(), "rendement moyen train = %.5f, écart-type = %.5f" % (data[a]["mu"], data[a]["sd"]))


def features(a):
    """Matrice (T, n_feat) de rendements standardisés ; colonne 0 = actif cible."""
    cols = [(data[a]["r"] - data[a]["mu"]) / data[a]["sd"]]
    if USE_OTHER_ASSET:
        b = "eth" if a == "btc" else "btc"
        cols.append((data[b]["r"] - data[b]["mu"]) / data[b]["sd"])
    return np.stack(cols, axis=1).astype(np.float32)


def make_windows(F, hist_len, L):
    """Fenêtres (X: L pas passés, y: H rendements futurs de l'actif cible) avec hist_len rendements connus."""
    idx = range(L, hist_len - H + 1)
    X = np.stack([F[i - L : i] for i in idx])
    y = np.stack([F[i : i + H, 0] for i in idx])
    return X, y

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. Modèles

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-10-10T11:23:40.058626Z","iopub.execute_input":"2026-10-10T11:23:40.059346Z","iopub.status.idle":"2026-10-10T11:23:40.071077Z","shell.execute_reply.started":"2026-10-10T11:23:40.059318Z","shell.execute_reply":"2026-10-10T11:23:40.070286Z"}}
class RNNNet(nn.Module):
    def __init__(self, kind, n_feat, units, layers, dropout):
        super().__init__()
        rnn = nn.LSTM if kind == "LSTM" else nn.GRU
        self.rnn = rnn(n_feat, units, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(units, H)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.drop(out[:, -1]))


def to_t(x):
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


def train_model(model, X, y, epochs, patience, lr, val_frac=VAL_FRAC):
    """Entraîne avec early stopping sur la queue chronologique (gap de H entre train et val)."""
    N = len(X)
    nv = max(int(N * val_frac), H + 1)
    tr_end = N - nv - H
    Xtr, ytr, Xva, yva = to_t(X[:tr_end]), to_t(y[:tr_end]), to_t(X[N - nv:]), to_t(y[N - nv:])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    best, best_state, wait = float("inf"), None, 0
    hist = {"train": [], "val": []}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEVICE)
        tot = 0.0
        for i in range(0, len(Xtr), BATCH):
            b = perm[i : i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            v = lossf(model(Xva), yva).item()
        hist["train"].append(tot / len(Xtr))
        hist["val"].append(v)
        if v < best - 1e-6:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return hist, (Xva, yva)


def new_model(kind, cfg, n_feat):
    return RNNNet(kind, n_feat, cfg["units"], cfg["layers"], cfg["dropout"]).to(DEVICE)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Time Series Cross-Validation (sélection des hyperparamètres, TRAIN uniquement)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-10-10T11:23:40.072052Z","iopub.execute_input":"2026-10-10T11:23:40.072239Z","iopub.status.idle":"2026-10-10T11:26:06.996420Z","shell.execute_reply.started":"2026-10-10T11:23:40.072224Z","shell.execute_reply":"2026-10-10T11:26:06.995520Z"}}
def grid_list():
    keys = list(GRID)
    out = [dict()]
    for k in keys:
        out = [{**o, k: v} for o in out for v in GRID[k]]
    return out


def cv_search(kind, a):
    F = features(a)
    rows = []
    for cfg in grid_list():
        X, y = make_windows(F, n_train - 1, cfg["lookback"])
        scores = []
        for tr, te in TimeSeriesSplit(n_splits=CV_SPLITS, gap=H).split(X):
            set_seed()
            m = new_model(kind, cfg, F.shape[1])
            train_model(m, X[tr], y[tr], EPOCHS_FULL, PATIENCE_FULL, LR)
            m.eval()
            with torch.no_grad():
                scores.append(nn.functional.mse_loss(m(to_t(X[te])), to_t(y[te])).item())
        rows.append({"actif": a.upper(), "modele": kind, **cfg, "batch": BATCH, "lr": LR,
                     "cv_mse_std": np.mean(scores), "cv_mse_sd": np.std(scores)})
        print(f"  CV {kind} {a.upper()} {cfg} -> {np.mean(scores):.4f}")
    return rows


best_cfg, cv_rows = {}, []
for a in ASSETS:
    for kind in MODELS:
        if RUN_CV:
            print(f"Recherche d'hyperparamètres : {kind} {a.upper()}")
            rows = cv_search(kind, a)
            cv_rows += rows
            b = min(rows, key=lambda r: r["cv_mse_std"])
            best_cfg[(kind, a)] = {k: b[k] for k in ["lookback", "units", "layers", "dropout"]}
        else:
            best_cfg[(kind, a)] = dict(DEFAULT_CFG)
if cv_rows:
    pd.DataFrame(cv_rows).to_csv(TAB / "05_cv_hyperparametres.csv", index=False)
pd.DataFrame([{"actif": a.upper(), "modele": k, **c, "batch": BATCH, "lr": LR}
              for (k, a), c in best_cfg.items()]).to_csv(TAB / "05_hyperparametres_retenus.csv", index=False)
print(pd.read_csv(TAB / "05_hyperparametres_retenus.csv"))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Walk-forward sur le test (mêmes origines que P1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-10-10T11:26:06.997660Z","iopub.execute_input":"2026-10-10T11:26:06.997946Z","iopub.status.idle":"2026-10-10T11:27:17.624754Z","shell.execute_reply.started":"2026-10-10T11:26:06.997920Z","shell.execute_reply":"2026-10-10T11:27:17.624133Z"}}
def walk_forward(kind, a, cfg):
    F, P, mu, sd = features(a), data[a]["P"], data[a]["mu"], data[a]["sd"]
    L = cfg["lookback"]
    set_seed()
    model, rows, curves = None, [], None
    t0 = time.time()
    for k, p in enumerate(origin_pos):
        hist_len = p - 1                       # rendements connus à l'origine p (dernier prix connu : P[p-1])
        X, y = make_windows(F, hist_len, L)
        if model is None:
            model = new_model(kind, cfg, F.shape[1])
            curves, val = train_model(model, X, y, EPOCHS_FULL, PATIENCE_FULL, LR)
        else:
            _, val = train_model(model, X, y, EPOCHS_FT, PATIENCE_FT, LR / 3)
        model.eval()
        with torch.no_grad():
            pred = model(to_t(F[hist_len - L : hist_len][None]))[0].cpu().numpy()
            pv = model(val[0]).cpu().numpy()
        cum = np.cumsum(pred * sd + mu)
        price = P[p - 1] * np.exp(cum)
        # IC 95 % : écart-type des erreurs de rendement cumulé sur la validation, par horizon
        res = np.cumsum(pv * sd + mu, axis=1) - np.cumsum(val[1].cpu().numpy() * sd + mu, axis=1)
        sig = res.std(axis=0)
        for h in range(H):
            rows.append({"origin": dates[p].strftime("%Y-%m-%d"), "date": dates[p + h].strftime("%Y-%m-%d"),
                         "h": h + 1, "y_true": P[p + h], "y_pred": price[h],
                         "lo95": price[h] * np.exp(-1.96 * sig[h]), "hi95": price[h] * np.exp(1.96 * sig[h])})
    fc = pd.DataFrame(rows)
    return fc, curves, time.time() - t0


forecasts, losses, times = {}, {}, {}
for a in ASSETS:
    for kind in MODELS:
        print(f"Walk-forward {kind} {a.upper()} avec {best_cfg[(kind, a)]}")
        fc, curves, t = walk_forward(kind, a, best_cfg[(kind, a)])
        forecasts[(kind.lower(), a)], losses[(kind, a)], times[(kind.lower(), a)] = fc, curves, t
        fc.to_csv(TAB / f"forecasts_{kind.lower()}_{a}.csv", index=False)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. Métriques (même définition que P1) et comparaison avec Naive / ARIMA

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-10-10T11:27:17.625879Z","iopub.execute_input":"2026-10-10T11:27:17.626230Z","iopub.status.idle":"2026-10-10T11:27:17.824293Z","shell.execute_reply.started":"2026-10-10T11:27:17.626209Z","shell.execute_reply":"2026-10-10T11:27:17.823718Z"}}
def metrics(fc):
    e = fc["y_true"] - fc["y_pred"]
    out = {"MSE": (e ** 2).mean(), "RMSE": np.sqrt((e ** 2).mean()), "MAE": e.abs().mean(),
           "MAPE_%": (e.abs() / fc["y_true"].abs()).mean() * 100}
    if fc["lo95"].notna().all():
        out["couverture_IC95"] = ((fc["y_true"] >= fc["lo95"]) & (fc["y_true"] <= fc["hi95"])).mean()
    else:
        out["couverture_IC95"] = np.nan
    return out


# baselines de P1 (si le dépôt est dans /kaggle/input) ; sinon Naive recalculé
base = {}
for a in ASSETS:
    for f in Path("/kaggle/input").rglob(f"forecasts_*_{a}.csv") if Path("/kaggle/input").exists() else []:
        name = f.stem[len("forecasts_"): -len(f"_{a}")]
        d = pd.read_csv(f)
        if {"origin", "h", "y_pred", "y_true"} <= set(d.columns) and name not in base.get(a, {}):
            d = d[d["origin"].isin([o for o in forecasts[("lstm", a)]["origin"].unique()])]
            base.setdefault(a, {})[name] = d
    if "naive" not in base.get(a, {}):
        P = data[a]["P"]
        rows = [{"origin": dates[p].strftime("%Y-%m-%d"), "h": h + 1, "y_true": P[p + h], "y_pred": P[p - 1],
                 "lo95": np.nan, "hi95": np.nan} for p in origin_pos for h in range(H)]
        base.setdefault(a, {})["naive"] = pd.DataFrame(rows)

glob_rows, hor_rows = [], []
for a in ASSETS:
    allm = {**{k: v for (k, aa), v in forecasts.items() if aa == a}, **base[a]}
    for name, fc in allm.items():
        if "lo95" not in fc:
            fc = fc.assign(lo95=np.nan, hi95=np.nan)
        glob_rows.append({"actif": a.upper(), "modele": name, **metrics(fc),
                          "temps_s": round(times.get((name, a), np.nan), 1)})
        for h in (1, 5, 10):
            hor_rows.append({"actif": a.upper(), "modele": name, "h": h, **metrics(fc[fc["h"] == h])})
res_glob = pd.DataFrame(glob_rows).round(4)
res_hor = pd.DataFrame(hor_rows).round(4)
res_glob.to_csv(TAB / "05_comparaison_modeles.csv", index=False)
res_glob[res_glob["modele"].isin(["lstm", "gru"])].to_csv(TAB / "05_metriques_dl.csv", index=False)
res_hor.to_csv(TAB / "05_metriques_dl_par_horizon.csv", index=False)
print(res_glob.to_string(index=False))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 6. Test de Diebold-Mariano (correction de Harvey, perte quadratique)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-10-10T11:27:17.825326Z","iopub.execute_input":"2026-10-10T11:27:17.825792Z","iopub.status.idle":"2026-10-10T11:27:17.973982Z","shell.execute_reply.started":"2026-10-10T11:27:17.825772Z","shell.execute_reply":"2026-10-10T11:27:17.973343Z"}}
def dm_test(e1, e2, lag=0, power=2):
    """H0 : même précision. Statistique < 0 => le modèle 1 est meilleur."""
    d = np.abs(e1) ** power - np.abs(e2) ** power
    m, dbar = len(d), d.mean()
    if np.allclose(d, 0):
        return np.nan, np.nan
    dc = d - dbar
    lrv = (dc ** 2).mean()
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] * dc[:-k]).mean()
    if lrv <= 0:
        lrv = (dc ** 2).mean()
    hh = lag + 1
    stat = dbar / np.sqrt(lrv / m) * np.sqrt((m + 1 - 2 * hh + hh * (hh - 1) / m) / m)
    return stat, 2 * (1 - stats.t.cdf(abs(stat), df=m - 1))


def errors(fc):
    return fc.set_index(["origin", "h"])["y_true"] - fc.set_index(["origin", "h"])["y_pred"]


dm_rows = []
for a in ASSETS:
    allm = {**{k: v for (k, aa), v in forecasts.items() if aa == a}, **base[a]}
    pairs = [("lstm", "gru"), ("lstm", "naive"), ("gru", "naive")]
    pairs += [(m, r) for m in ("lstm", "gru") for r in allm if r.startswith("arima")]
    for m1, m2 in pairs:
        e1, e2 = errors(allm[m1]), errors(allm[m2])
        idx = e1.index.intersection(e2.index).sort_values()
        e1, e2 = e1.loc[idx], e2.loc[idx]
        s, p = dm_test(e1.values, e2.values, lag=H - 1)
        dm_rows.append({"actif": a.upper(), "modele1": m1, "modele2": m2, "h": "tous (1-10)",
                        "DM_stat": s, "p_value": p, "n": len(idx)})
        for h in (1, 5, 10):
            s, p = dm_test(e1.xs(h, level="h").values, e2.xs(h, level="h").values, lag=0)
            dm_rows.append({"actif": a.upper(), "modele1": m1, "modele2": m2, "h": h,
                            "DM_stat": s, "p_value": p, "n": int((idx.get_level_values("h") == h).sum())})
dm = pd.DataFrame(dm_rows).round(4)
dm["conclusion_5%"] = np.where(dm["p_value"] >= 0.05, "pas de différence significative",
                               np.where(dm["DM_stat"] < 0, "modèle 1 meilleur", "modèle 2 meilleur"))
dm.to_csv(TAB / "05_dm_tests.csv", index=False)
print(dm.to_string(index=False))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 7. Figures

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-10-10T11:27:17.975052Z","iopub.execute_input":"2026-10-10T11:27:17.975803Z","iopub.status.idle":"2026-10-10T11:27:20.111487Z","shell.execute_reply.started":"2026-10-10T11:27:17.975779Z","shell.execute_reply":"2026-10-10T11:27:20.110722Z"}}
# 7.1 courbes de perte train / validation (entraînement initial)
fig, axes = plt.subplots(len(ASSETS), len(MODELS), figsize=(11, 7))
for i, a in enumerate(ASSETS):
    for j, kind in enumerate(MODELS):
        ax, c = axes[i, j], losses[(kind, a)]
        ax.plot(c["train"], label="train")
        ax.plot(c["val"], label="validation")
        ax.set_title(f"{kind} — {a.upper()}")
        ax.set_xlabel("époque")
        ax.set_ylabel("MSE (rendements standardisés)")
        ax.legend()
plt.tight_layout()
plt.savefig(FIG / "05_courbes_perte.png", dpi=130)
plt.close()

# 7.2 prévisions 10 pas sur le test
fig, axes = plt.subplots(len(ASSETS), 1, figsize=(12, 8))
for ax, a in zip(axes, ASSETS):
    P = data[a]["P"]
    ax.plot(dates[n_train:], P[n_train:], color="black", lw=1, label="réel")
    for kind, col in zip(MODELS, ["tab:blue", "tab:red"]):
        fc = forecasts[(kind.lower(), a)]
        for k, (o, g) in enumerate(fc.groupby("origin")):
            ax.plot(pd.to_datetime(g["date"]), g["y_pred"], color=col, lw=1.3, label=kind if k == 0 else None)
    ax.set_title(f"{a.upper()} — prévisions à 10 jours (une origine tous les 10 jours)")
    ax.legend()
plt.tight_layout()
plt.savefig(FIG / "05_previsions_lstm_gru.png", dpi=130)
plt.close()

# 7.3 RMSE par horizon
fig, axes = plt.subplots(1, len(ASSETS), figsize=(12, 4))
for ax, a in zip(axes, ASSETS):
    allm = {**{k: v for (k, aa), v in forecasts.items() if aa == a}, **base[a]}
    for name, fc in allm.items():
        g = fc.assign(e2=(fc["y_true"] - fc["y_pred"]) ** 2).groupby("h")["e2"].mean() ** 0.5
        ax.plot(g.index, g.values, marker="o", label=name)
    ax.set_title(f"RMSE par horizon — {a.upper()}")
    ax.set_xlabel("horizon h (jours)")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG / "05_rmse_par_horizon.png", dpi=130)
plt.close()

print("\nTerminé. Fichiers dans :", OUT / "results")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-10T11:27:20.113431Z","iopub.execute_input":"2026-10-10T11:27:20.113747Z","iopub.status.idle":"2026-10-10T11:27:20.145169Z","shell.execute_reply.started":"2026-10-10T11:27:20.113729Z","shell.execute_reply":"2026-10-10T11:27:20.144606Z"},"jupyter":{"outputs_hidden":false}}
import shutil
import os

results_dir = "/kaggle/working/results"

if os.path.exists(results_dir):
    shutil.make_archive(
        "/kaggle/working/results",
        "zip",
        results_dir
    )
    print("Fichier créé : /kaggle/working/results.zip")
else:
    print("Le dossier results/ n'existe pas.")