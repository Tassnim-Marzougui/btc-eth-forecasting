"""Application Streamlit — BTC & ETH : cointégration, volatilité et Deep Learning.

Lancer depuis la racine du dépôt :   streamlit run app/streamlit_app.py
L'app ne recalcule rien : elle lit les données (data/), les tableaux et prévisions (results/tables/),
les figures (results/figures/) et les textes LLM (notebooks/03_DeepLearning_LLM/) déjà produits par le projet.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "data").exists() else HERE
DATA, TAB, FIG = ROOT / "data", ROOT / "results" / "tables", ROOT / "results" / "figures"
LLM_DIR = ROOT / "notebooks" / "03_DeepLearning_LLM"

st.set_page_config(page_title="BTC & ETH — Séries temporelles", page_icon="📈", layout="wide")

LABELS = {"naive": "Naive (marche aléatoire)", "drift": "Drift", "arima_aic": "ARIMA (AIC)",
          "arima_bic": "ARIMA (BIC) = Naive", "var": "VAR", "vecm": "VECM", "lstm": "LSTM", "gru": "GRU"}
COLORS = {"naive": "#7f7f7f", "drift": "#bcbd22", "arima_aic": "#1f77b4", "arima_bic": "#17becf",
          "var": "#2ca02c", "vecm": "#d62728", "lstm": "#9467bd", "gru": "#ff7f0e"}


# ------------------------------------------------------------------ chargement
@st.cache_data
def table(name):
    p = TAB / name
    return pd.read_csv(p) if p.exists() else None


def show_table(name, caption=None):
    df = table(name)
    if df is None:
        st.info(f"Fichier introuvable : results/tables/{name}")
        return
    if caption:
        st.caption(caption)
    st.dataframe(df, hide_index=True)


def show_fig(name, caption=None):
    p = FIG / name
    if p.exists():
        st.image(str(p), caption=caption)
    else:
        st.info(f"Figure introuvable : results/figures/{name}")


def read_text(path):
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else None


@st.cache_data
def prices():
    p = DATA / "btc_eth_processed.csv"
    if not p.exists():
        p = DATA / "btc_eth.csv"
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    for a in ("btc", "eth"):
        df[f"{a}_ret"] = np.log(df[f"{a}_close"]).diff()
        df[f"{a}_vol30"] = df[f"{a}_ret"].rolling(30).std() * np.sqrt(365)
        df[f"{a}_ma50"] = df[f"{a}_close"].rolling(50).mean()
        df[f"{a}_ma200"] = df[f"{a}_close"].rolling(200).mean()
    return df


@st.cache_data
def test_start():
    import json
    p = DATA / "split.json"
    return json.loads(p.read_text())["test_start"] if p.exists() else None


@st.cache_data
def forecasts(asset):
    """Prévisions de prix de tous les modèles, au format commun origin, date, h, y_true, y_pred, lo95, hi95."""
    out = {}
    for m in ("naive", "drift", "arima_aic", "arima_bic", "lstm", "gru"):
        p = TAB / f"forecasts_{m}_{asset}.csv"
        if p.exists():
            d = pd.read_csv(p)
            for c in ("lo95", "hi95"):
                if c not in d:
                    d[c] = np.nan
            out[m] = d[["origin", "date", "h", "y_true", "y_pred", "lo95", "hi95"]]
    p2, raw = TAB / "forecasts_P2.csv", DATA / "btc_eth.csv"
    if p2.exists() and raw.exists():
        dates = pd.read_csv(raw, usecols=["date"])["date"]
        d = pd.read_csv(p2)
        d = d[d.asset == asset.upper()].copy()
        d["origin"] = d["origin"].map(lambda i: dates[int(i)])
        for m in ("VAR", "VECM"):
            x = d[d.model == m][["origin", "date", "h", "y_true", "y_pred"]].copy()
            x["lo95"], x["hi95"] = np.nan, np.nan
            out[m.lower()] = x
    return out


def ts(x):
    return pd.Timestamp(x).strftime("%Y-%m-%d")


# ------------------------------------------------------------------ pages
def page_home():
    st.title("BTC & ETH : cointégration, volatilité et Deep Learning")
    st.markdown("**Question.** BTC et ETH partagent-ils un équilibre de long terme, et un modèle économétrique "
                "ou neuronal bat-il *significativement* la marche aléatoire pour prévoir le prix à 10 jours ?")
    px_ = prices()
    f = table("07_tableau_final.csv")
    c = st.columns(4)
    c[0].metric("Observations", f"{len(px_):,}".replace(",", " "), f"{px_.index.min():%Y} → {px_.index.max():%Y}")
    c[1].metric("Jours de test", 320, "15/11/2025 → 30/09/2026")
    c[2].metric("Origines de prévision", 32, "horizon 10 jours")
    c[3].metric("Modèles comparés", len(LABELS))
    st.success("**Réponse courte : non.** Aucun modèle (ARIMA, VAR, VECM, LSTM, GRU) ne bat significativement la "
               "marche aléatoire (Diebold-Mariano, 5 %). Pas de cointégration BTC–ETH, et le VECM est "
               "significativement moins bon que Naive. La volatilité, elle, est modélisable (GARCH).")
    if f is not None:
        cols = st.columns(2)
        for col, a in zip(cols, ("BTC", "ETH")):
            d = f[f.actif == a].sort_values("RMSE")
            col.subheader(f"{a} — RMSE par modèle (USD)")
            fig = go.Figure(go.Bar(x=d.RMSE, y=[LABELS.get(m, m) for m in d.modele], orientation="h",
                                   marker_color=[COLORS.get(m) for m in d.modele],
                                   text=d.RMSE.round(1), textposition="outside"))
            fig.update_layout(height=340, margin=dict(l=0, r=30, t=10, b=0), yaxis=dict(autorange="reversed"),
                              xaxis=dict(range=[d.RMSE.min() * 0.9, d.RMSE.max() * 1.05]))
            col.plotly_chart(fig)
        st.caption("L'axe commence à 90 % du meilleur RMSE pour rendre les écarts lisibles : ils restent petits "
                   "et ne sont pas statistiquement significatifs (page « Comparaison finale »).")
    st.info("Utilise le menu de gauche pour explorer les données, les modèles, les prévisions et l'analyse du LLM.")


def page_data():
    st.header("Données et analyse exploratoire")
    a = st.radio("Actif", ["btc", "eth"], horizontal=True, format_func=str.upper)
    px_, t0 = prices(), test_start()
    log = st.checkbox("Échelle logarithmique", value=True)
    show_ma = st.checkbox("Moyennes mobiles 50 j et 200 j", value=True)
    fig = go.Figure(go.Scatter(x=px_.index, y=px_[f"{a}_close"], name="Prix", line=dict(width=1.4)))
    if show_ma:
        fig.add_trace(go.Scatter(x=px_.index, y=px_[f"{a}_ma50"], name="MA 50", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=px_.index, y=px_[f"{a}_ma200"], name="MA 200", line=dict(width=1)))
    if t0:
        fig.add_shape(type="rect", xref="x", yref="paper", x0=ts(t0), x1=ts(px_.index.max()), y0=0, y1=1,
                      fillcolor="rgba(255,160,0,0.15)", line_width=0)
        fig.add_annotation(x=ts(t0), y=1, yref="paper", text="période de test", showarrow=False, xanchor="left")
    fig.update_layout(height=420, yaxis_type="log" if log else "linear", yaxis_title="USD",
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)
    c1, c2 = st.columns(2)
    v = go.Figure(go.Scatter(x=px_.index, y=px_[f"{a}_vol30"], line=dict(width=1)))
    v.update_layout(title="Volatilité réalisée 30 jours (annualisée)", height=300, margin=dict(l=0, r=0, t=40, b=0))
    c1.plotly_chart(v)
    h = go.Figure(go.Histogram(x=px_[f"{a}_ret"].dropna(), nbinsx=120))
    h.update_layout(title="Distribution des log-rendements (queues épaisses)", height=300,
                    margin=dict(l=0, r=0, t=40, b=0))
    c2.plotly_chart(h)
    rc = px_["btc_ret"].rolling(90).corr(px_["eth_ret"])
    r = go.Figure(go.Scatter(x=rc.index, y=rc, line=dict(width=1.2)))
    r.update_layout(title="Corrélation glissante 90 jours BTC–ETH (rendements)", height=300,
                    margin=dict(l=0, r=0, t=40, b=0), yaxis_range=[0, 1])
    st.plotly_chart(r)
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Statistiques descriptives (log-rendements)")
        show_table("01_stats_descriptives.csv")
    with c4:
        st.subheader("Stationnarité (ADF / KPSS)")
        show_table("01_stationnarite.csv")
    st.caption("Lecture : les log-prix sont non stationnaires, les rendements stationnaires (d = 1). "
               "Kurtosis élevé et Jarque-Bera rejeté : rendements non gaussiens, queues épaisses.")


def page_models():
    st.header("Modèles économétriques (P1 et P2)")
    t1, t2, t3 = st.tabs(["ARIMA / SARIMA", "VAR · Granger · Cointégration", "GARCH"])
    with t1:
        st.subheader("Sélection des ordres")
        show_table("02_selection_ordres.csv")
        st.subheader("Diagnostic des résidus")
        show_table("02_diagnostic_residus.csv", "Ljung-Box, Jarque-Bera, Shapiro, ARCH-LM (p-values)")
        c = st.columns(2)
        with c[0]:
            show_fig("02_previsions_btc_arima_aic.png", "Prévisions ARIMA (AIC) — BTC")
        with c[1]:
            show_fig("02_residus_btc_AIC.png", "Résidus ARIMA (AIC) — BTC")
    with t2:
        c = st.columns(3)
        with c[0]:
            st.subheader("Granger")
            show_table("03_granger.csv", "Causalité prédictive, pas économique")
        with c[1]:
            st.subheader("Johansen")
            show_table("03_cointegration.csv", "Trace < seuil 95 % : pas de cointégration")
        with c[2]:
            st.subheader("Engle-Granger")
            show_table("03_engle_granger.csv", "p élevée : pas de cointégration")
        st.subheader("Métriques VAR / VECM")
        show_table("03_metriques_var.csv")
        c = st.columns(2)
        with c[0]:
            show_fig("03_correlation_glissante.png")
        with c[1]:
            show_fig("03_irf.png")
        st.warning("Johansen et Engle-Granger ne détectent aucune cointégration : le VECM n'est pas justifié "
                   "et il est effectivement moins bon que Naive. Il est présenté pour la démonstration.")
    with t3:
        c = st.columns(2)
        with c[0]:
            st.subheader("Effets ARCH")
            show_table("04_arch_lm.csv")
            st.subheader("Diagnostic GARCH")
            show_table("04_diagnostic_garch.csv")
        with c[1]:
            st.subheader("Comparaison GARCH / asymétrie")
            show_table("04_garch_asymetrique.csv")
            st.subheader("Prévision de volatilité (QLIKE)")
            show_table("04_metriques_garch.csv")
        c = st.columns(2)
        with c[0]:
            show_fig("04_volatilite_conditionnelle.png")
        with c[1]:
            show_fig("04_var_risk.png")


def page_dl():
    st.header("Deep Learning : LSTM et GRU (P3)")
    st.markdown("Entrées : log-rendements standardisés (statistiques du *train* seulement). Sortie : les 10 "
                "rendements suivants, reconvertis en prix. Hyperparamètres choisis par *Time Series "
                "Cross-Validation* sur le train ; évaluation en *walk-forward* sur les 32 origines.")
    st.subheader("Hyperparamètres retenus")
    show_table("05_hyperparametres_retenus.csv", "Adam, batch 64, learning rate 1e-3")
    cv = table("05_cv_hyperparametres.csv")
    if cv is not None:
        st.subheader("Validation croisée temporelle : meilleures configurations")
        top = cv.sort_values("cv_mse_std").groupby(["actif", "modele"]).head(3)
        st.dataframe(top.round(4), hide_index=True)
        st.caption("Les scores diffèrent très peu entre configurations : l'écart-type entre plis (~0,39) est bien "
                   "supérieur aux écarts de moyenne. La grille ne départage pas vraiment les modèles.")
    c = st.columns(2)
    with c[0]:
        show_fig("05_courbes_perte.png", "Courbes de perte : plates autour de MSE ≈ 1 (variance des rendements)")
    with c[1]:
        show_fig("05_previsions_lstm_gru.png", "Prévisions à 10 jours : proches de Naive + légère dérive")
    st.subheader("Métriques par horizon")
    show_table("05_metriques_dl_par_horizon.csv")
    st.info("Interprétation : les réseaux ne trouvent pas de signal prévisible dans les rendements à 1-10 jours ; "
            "leurs prévisions sont quasi identiques à la marche aléatoire.")


def page_compare():
    st.header("Comparaison finale de tous les modèles")
    f, hz = table("07_tableau_final.csv"), table("07_metriques_par_horizon.csv")
    if f is None:
        st.info("Lance d'abord src/deep_learning/07_tableau_final.py")
        return
    a = st.radio("Actif", ["BTC", "ETH"], horizontal=True)
    d = f[f.actif == a].sort_values("RMSE")
    st.subheader("Tableau unique (prix en USD, 32 origines, h = 1 à 10)")
    d2 = d.assign(modèle=d.modele.map(LABELS)).drop(columns=["actif", "modele", "n"])
    st.dataframe(d2[["modèle"] + [c for c in d2.columns if c != "modèle"]], hide_index=True)
    st.caption("GARCH n'apparaît pas ici : il prévoit la volatilité, pas le prix (voir « Modèles économétriques »).")
    if hz is not None:
        st.subheader("Erreur par horizon")
        mode = st.radio("Affichage", ["RMSE brut", "Ratio vs Naive"], horizontal=True)
        h = hz[hz.actif == a].copy()
        if mode == "Ratio vs Naive":
            base = h[h.modele == "naive"].set_index("h")["RMSE"]
            h["y"] = h.apply(lambda r: r.RMSE / base[r.h], axis=1)
        else:
            h["y"] = h["RMSE"]
        fig = go.Figure()
        for m in h.modele.unique():
            if m == "arima_bic":
                continue
            g = h[h.modele == m].sort_values("h")
            fig.add_trace(go.Scatter(x=g.h, y=g.y, name=LABELS.get(m, m), mode="lines+markers",
                                     line=dict(color=COLORS.get(m))))
        fig.update_layout(height=380, xaxis=dict(tickvals=[1, 5, 10], title="horizon h (jours)"),
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
    st.subheader("Test de Diebold-Mariano (p-values, erreur quadratique, correction de Harvey)")
    M = table("07_dm_matrice_pvalues.csv")
    if M is not None:
        m = M[M.actif == a].drop(columns="actif").set_index("modele1")
        z = m.values.astype(float)
        txt = [[("" if np.isnan(v) else f"{v:.3f}") for v in row] for row in z]
        hm = go.Figure(go.Heatmap(z=z, x=[LABELS.get(c, c) for c in m.columns], y=[LABELS.get(i, i) for i in m.index],
                                  text=txt, texttemplate="%{text}", colorscale="Blues_r", zmin=0, zmax=0.1,
                                  colorbar=dict(title="p-value")))
        hm.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"))
        st.plotly_chart(hm)
        st.caption("Case foncée (p < 0,05) = différence significative entre les deux modèles. "
                   "Seuls le VECM (et le VAR sur BTC) sont significativement moins bons que Naive. "
                   "Cases vides : prévisions identiques (ARIMA BIC = Naive).")
    dm = table("07_dm_vs_naive.csv")
    if dm is not None:
        with st.expander("Détail : chaque modèle contre Naive (global et par horizon)"):
            st.dataframe(dm[dm.actif == a], hide_index=True)


def page_forecast():
    st.header("Prévisions interactives")
    a = st.radio("Actif", ["btc", "eth"], horizontal=True, format_func=str.upper)
    fc = forecasts(a)
    if not fc:
        st.info("Aucune prévision trouvée dans results/tables/.")
        return
    avail = [m for m in LABELS if m in fc]
    chosen = st.multiselect("Modèles à comparer", avail, format_func=LABELS.get,
                            default=[m for m in ("naive", "arima_aic", "lstm", "gru") if m in avail])
    origins = sorted(fc[avail[0]].origin.unique())
    o = st.select_slider("Origine de prévision (premier jour prévu)", options=origins, value=origins[0])
    px_ = prices()
    close = px_[f"{a.lower()}_close"]
    o_ts = pd.Timestamp(o)
    hist = close[o_ts - pd.Timedelta(days=45): o_ts - pd.Timedelta(days=1)]
    real = fc[avail[0]].query("origin == @o").sort_values("h")
    last = (hist.index[-1], hist.iloc[-1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Historique", line=dict(color="black", width=1.5)))
    fig.add_trace(go.Scatter(x=pd.to_datetime(real.date), y=real.y_true, name="Réel", mode="lines+markers",
                             line=dict(color="black", dash="dot")))
    band_done = False
    for m in chosen:
        g = fc[m].query("origin == @o").sort_values("h")
        x = [last[0]] + list(pd.to_datetime(g.date))
        fig.add_trace(go.Scatter(x=x, y=[last[1]] + list(g.y_pred), name=LABELS[m], mode="lines+markers",
                                 line=dict(color=COLORS.get(m))))
        if not band_done and g.lo95.notna().all():
            xd = list(pd.to_datetime(g.date))
            fig.add_trace(go.Scatter(x=xd, y=g.hi95, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=xd, y=g.lo95, line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(120,120,120,0.2)", name=f"IC 95 % ({LABELS[m]})",
                                     hoverinfo="skip"))
            band_done = True
    fig.update_layout(height=470, yaxis_title="USD", margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig)
    rows = []
    for m in chosen:
        g, allm = fc[m].query("origin == @o"), fc[m]
        e, ea = g.y_true - g.y_pred, allm.y_true - allm.y_pred
        rows.append({"Modèle": LABELS[m], "RMSE (cette origine)": round(float(np.sqrt((e ** 2).mean())), 2),
                     "MAPE % (cette origine)": round(float((e.abs() / g.y_true).mean() * 100), 2),
                     "RMSE (32 origines)": round(float(np.sqrt((ea ** 2).mean())), 2)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Sur une origine isolée, un modèle peut sembler meilleur par hasard : seul le test de "
               "Diebold-Mariano sur l'ensemble des origines permet de conclure.")


def page_llm():
    st.header("IA générative : analyse critique de Grok (P3)")
    st.markdown("Protocole : 3 prompts (hypothèses, vulgarisation, recommandations simulées), puis vérification "
                "automatique des réponses contre nos sorties (`06_llm_analysis.py`) et comparaison avec "
                "l'interprétation de l'équipe.")
    t1, t2, t3 = st.tabs(["Interprétation de l'équipe", "Réponses de Grok", "Vérification automatique"])
    with t1:
        txt = read_text(LLM_DIR / "interpretation_equipe.md")
        st.markdown(txt) if txt else st.info("notebooks/03_DeepLearning_LLM/interpretation_equipe.md introuvable")
    with t2:
        txt = read_text(LLM_DIR / "grok_reponses.md")
        if txt:
            with st.expander("Prompts et réponses intégraux", expanded=False):
                st.text(txt)
        else:
            st.info("notebooks/03_DeepLearning_LLM/grok_reponses.md introuvable")
    with t3:
        st.subheader("Synthèse")
        show_table("06_synthese_llm.csv")
        st.subheader("Nombres cités par le LLM contre nos tableaux")
        v = table("06_verification_chiffres.csv")
        if v is not None:
            st.metric("Nombres non retrouvés dans nos sorties", int((v.statut == "NON TROUVÉ").sum()),
                      f"sur {len(v)} nombres vérifiés")
            st.dataframe(v, hide_index=True)
            st.caption("« Vérifié » = le nombre existe dans nos tableaux ; il faut encore relire le contexte.")
        st.subheader("Formulations à risque")
        show_table("06_alertes_formulation.csv")


PAGES = {"Accueil": page_home, "Données et EDA": page_data, "Modèles économétriques": page_models,
         "Deep Learning": page_dl, "Comparaison finale": page_compare,
         "Prévisions interactives": page_forecast, "Analyse du LLM": page_llm}

st.sidebar.title("BTC & ETH")
choice = st.sidebar.radio("Navigation", list(PAGES))
st.sidebar.caption("Mini-projet Séries Temporelles — Tek-Up, 2026")
PAGES[choice]()