"""06 — Analyse critique des réponses d'un LLM (Grok) : vérification automatique contre nos sorties.

Usage (à la racine du dépôt) :
    python src/deep_learning/06_llm_analysis.py --file notebooks/03_DeepLearning_LLM/grok_reponses.md

Le fichier contient les prompts ET les réponses collés tels quels. Le script isole les réponses
(il ignore les prompts, dont les chiffres viennent de nous) puis :
  1. extrait chaque nombre cité par le LLM et le cherche dans results/tables/*.csv
     -> "vérifié" (avec la source) ou "NON TROUVÉ" (à contrôler à la main : erreur ou invention ?)
  2. repère les formulations à risque (causalité tirée de Granger, "bat le marché" sans test DM,
     certitudes, extrapolation) et compte les "hypothèse non vérifiée" (consigne du prompt)
  3. contrôle les contraintes de longueur (vulgarisation <= 200 mots)
Sorties : results/tables/06_verification_chiffres.csv, 06_alertes_formulation.csv, 06_synthese_llm.csv
Le script ne juge pas à la place de l'équipe : il prépare le tableau "humain vs LLM".
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

TAB = Path("results/tables")
CANDIDATES = [
    "notebooks/03_DeepLearning_LLM/grok_reponses.md",
    "notebooks/03_DeepLearning_LLM/grok_reponses.txt",
    "llm/grok_reponses.md",
    "llm/grok_reponses.txt",
]
# constantes du protocole (taille d'échantillon, niveaux de test...) : pas besoin de les chercher dans les tableaux
CONSTANTS = {0.05, 0.01, 0.1, 5, 10, 95, 99, 100, 30, 32, 320, 365, 2875, 3195, 2018, 2025, 2026}


# ------------------------------------------------------------------ 1. isoler les réponses
def split_responses(text, keep_all=False):
    """Retourne [(nom_segment, texte)] sans les prompts. Marqueurs : 'Réflexion' (Grok), 'TÂCHE', 'Relance N'."""
    lines = text.splitlines()
    if keep_all:
        return [("fichier complet", "\n".join(lines))]
    start = 0
    for i, l in enumerate(lines):
        if re.match(r"^\s*R[ée]flexion", l):
            start = i + 1
            break
    else:
        for i, l in enumerate(lines):
            if l.strip().startswith("TÂCHE"):
                start = i + 1
                break
    segs, name, buf, skip_prompt = [], "Prompt 1", [], False
    for l in lines[start:]:
        m = re.match(r"^\s*Relance\s*(\d)", l)
        if m:
            segs.append((name, "\n".join(buf)))
            name, buf, skip_prompt = f"Relance {m.group(1)}", [], True
            continue
        if skip_prompt and l.strip():          # première ligne non vide après "Relance N" = le prompt envoyé
            skip_prompt = False
            continue
        if " - Grok" in l or l.strip().startswith("Tu es un statisticien"):  # titre d'onglet copié par erreur
            continue
        buf.append(l)
    segs.append((name, "\n".join(buf)))
    return [(n, t) for n, t in segs if t.strip()]


# ------------------------------------------------------------------ 2. valeurs de référence
def load_reference():
    vals, src = [], []
    for f in sorted(TAB.glob("*.csv")):
        if f.name.startswith("forecasts_") or f.name.startswith("06_"):
            continue
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        for c in d.columns:
            col = pd.to_numeric(d[c], errors="coerce").dropna()
            for v in col:
                vals.append(float(v))
                src.append(f"{f.name}:{c}")
    return np.array(vals), np.array(src)


NUM = re.compile(r"(?<![\w.])[-−–]?\d+(?:[ \u202f\u00a0]\d{3})*(?:[.,]\d+)?(?:\s?[eE][-−+]?\d+)?")


def clean(text):
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{4}", " ", text)                      # dates
    text = re.sub(r"\(\s*\d+(?:\s*,\s*\d+)+\s*\)", " ", text)                  # ordres ARIMA/GARCH (1,1,0)
    return text


def to_float(tok):
    t = re.sub(r"[ \u202f\u00a0]", "", tok.replace("−", "-").replace("–", "-")).replace(",", ".")
    return float(t)


def decimals(tok):
    mant = re.split(r"[eE]", tok)[0]
    m = re.search(r"[.,](\d+)", mant)
    return len(m.group(1)) if m else 0


def sig_round(a, n):
    a = np.asarray(a, dtype=float)
    out = np.zeros_like(a)
    nz = a != 0
    scale = 10.0 ** (n - 1 - np.floor(np.log10(np.abs(a[nz]))))
    out[nz] = np.round(a[nz] * scale) / scale
    return out


def find_match(tok, V, S):
    x = abs(to_float(tok))
    is_sci = bool(re.search(r"[eE]", tok)) or (0 < x < 1e-3)
    d = decimals(tok)
    for arr, tag in ((np.abs(V), ""), (np.abs(V) * 100, " (×100 = %)"), (np.abs(V) / 100, " (÷100)")):
        if is_sci:
            mant = re.split(r"[eE]", tok)[0].replace("-", "").replace(",", ".").replace(".", "").lstrip("0")
            nsig = max(len(mant), 1)
            hit = np.where(np.isclose(sig_round(arr, nsig), x, rtol=1e-9, atol=0))[0]
        else:
            hit = np.where(np.isclose(np.round(arr, d), x, rtol=0, atol=1e-9))[0]
        if len(hit):
            return S[hit[0]] + tag
    return None


def check_numbers(segments, V, S):
    rows = []
    for name, text in segments:
        text = clean(text)
        for m in NUM.finditer(text):
            tok = m.group(0).strip()
            try:
                x = abs(to_float(tok))
            except ValueError:
                continue
            if x in CONSTANTS or (x == int(x) and x <= 12 and not re.search(r"[eE]", tok)):
                continue                                   # numérotation de listes, petits entiers, constantes
            ctx = text[max(0, m.start() - 45): m.end() + 30].replace("\n", " ")
            src = find_match(tok, V, S)
            rows.append({"segment": name, "nombre_cité": tok, "statut": "vérifié" if src else "NON TROUVÉ",
                         "source": src or "", "contexte": ctx})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3. formulations à risque
PATTERNS = [
    (r"\b(cause[sn]?|provoque[nt]?|entra[iî]ne[nt]?|m[eè]ne[nt]?|conduit|pilote[nt]?|leads?)\b",
     "causalité ? Granger = causalité prédictive, pas économique"),
    (r"\b(bat(?:tent)?|surperform\w*|domine[nt]?)\b", "affirmation de victoire : doit être appuyée par un test DM"),
    (r"\b(meilleurs?|meilleure?s?|supérieur\w*|plus performant\w*)\b",
     "classement : vérifier la significativité (Diebold-Mariano) avant de conclure"),
    (r"\b(prouve|démontre|garanti\w*|certain\w*|sûr\w*|sans aucun doute)\b", "certitude excessive"),
    (r"\b(va (?:monter|baisser|augmenter|chuter)|prévoit une (?:hausse|baisse)|achetez|vendez)\b",
     "extrapolation / conseil directif au-delà de ce que montrent les modèles"),
    (r"\b(apprentissage insuffisant|sous-entra[iî]n\w*|trop peu d['’]époques)\b",
     "interprétation des courbes plates à discuter : absence de signal vs entraînement insuffisant"),
]


def alerts(segments):
    rows = []
    for name, text in segments:
        for line in text.splitlines():
            for pat, why in PATTERNS:
                if re.search(pat, line, flags=re.I):
                    rows.append({"segment": name, "alerte": why, "extrait": line.strip()[:220]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="fichier texte/markdown avec prompts + réponses du LLM")
    ap.add_argument("--all", action="store_true", help="ne pas isoler les réponses : analyser tout le fichier")
    a = ap.parse_args()
    path = Path(a.file) if a.file else next((Path(c) for c in CANDIDATES if Path(c).exists()), None)
    if path is None or not path.exists():
        raise SystemExit("Fichier introuvable : utilise --file chemin/vers/grok_reponses.md")
    text = path.read_text(encoding="utf-8", errors="ignore")
    segs = split_responses(text, keep_all=a.all)
    V, S = load_reference()
    print(f"{path} | segments : {[n for n, _ in segs]} | {len(V)} valeurs de référence")

    nums = check_numbers(segs, V, S)
    nums.to_csv(TAB / "06_verification_chiffres.csv", index=False)
    al = alerts(segs)
    al.to_csv(TAB / "06_alertes_formulation.csv", index=False)

    syn = []
    for n, t in segs:
        k = nums[nums.segment == n] if len(nums) else nums
        syn.append({"segment": n, "mots": len(t.split()),
                    "nombres_cités": len(k), "vérifiés": int((k.statut == "vérifié").sum()) if len(k) else 0,
                    "non_trouvés": int((k.statut == "NON TROUVÉ").sum()) if len(k) else 0,
                    "alertes": int((al.segment == n).sum()) if len(al) else 0,
                    "mentions_hypothèse_non_vérifiée": len(re.findall(r"hypoth[eè]se non v[ée]rifi[ée]e", t, flags=re.I))})
    syn = pd.DataFrame(syn)
    syn.to_csv(TAB / "06_synthese_llm.csv", index=False)

    print("\n=== SYNTHÈSE ===")
    print(syn.to_string(index=False))
    r2 = syn[syn.segment == "Relance 2"]
    if len(r2):
        w = int(r2.mots.iloc[0])
        print(f"\nVulgarisation : {w} mots ({'OK' if w <= 200 else 'DÉPASSE'} la limite de 200).")
    if syn["mentions_hypothèse_non_vérifiée"].sum() == 0:
        print("Le LLM n'a écrit « hypothèse non vérifiée » nulle part alors que la consigne l'exigeait.")
    nf = nums[nums.statut == "NON TROUVÉ"] if len(nums) else nums
    print(f"\n=== NOMBRES NON TROUVÉS ({len(nf)}) : à vérifier à la main ===")
    for _, r in nf.iterrows():
        print(f"  [{r.segment}] {r['nombre_cité']:>10}  ...{r.contexte}...")
    print(f"\n=== ALERTES DE FORMULATION ({len(al)}) ===")
    for _, r in al.iterrows():
        print(f"  [{r.segment}] {r.alerte}\n      -> {r.extrait}")
    print("\nFichiers écrits dans", TAB)


if __name__ == "__main__":
    main()
