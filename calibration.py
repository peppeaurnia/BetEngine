"""
🎯 CALIBRATION — Calibrazione empirica data-driven
====================================================
Stima la relazione monotòna tra probabilità dichiarata dal modello e
frequenza reale osservata, usando una regressione isotonica (Pool Adjacent
Violators, implementata a mano: nessuna dipendenza da sklearn).

Se il modello storicamente dice 68% ma vince il 61%, un nuovo 68% diventa 61%.

v5 — DUE CORREZIONI IMPORTANTI

1. IL FIT ORA USA `prob_raw`, NON `prob`.
   Bug della v4: l'app salvava in `prob` la probabilità GIÀ calibrata, e il
   fit girava su quella colonna. Finché la calibrazione era inattiva non si
   notava nulla (apply() era l'identità), ma al primo refit dopo
   l'attivazione la curva veniva stimata su valori già corretti e poi
   riapplicata alle probabilità grezze: la correzione si componeva su sé
   stessa a ogni ricalcolo, allontanandosi progressivamente dai dati.
   Ora `prob_raw` conserva la probabilità ancorata ma NON calibrata, che è
   l'unico input corretto per il fit.

2. IL FIT USA TUTTI I PRONOSTICI, NON SOLO QUELLI CONSIGLIATI.
   La v4 salvava solo i 2-3 consigliati, tutti sopra le soglie: la curva
   poteva essere stimata solo nella fascia alta e non c'era modo di sapere
   se il modello fosse calibrato altrove. Ora l'app salva ogni mercato di
   ogni partita analizzata (`shortlisted` distingue i consigliati), quindi
   la curva copre l'intero range e il campione cresce 3-4 volte più in fretta.

Finché il campione è insufficiente, `apply()` restituisce la probabilità
invariata: il comportamento dell'app non cambia finché non ci sono dati.
"""

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np

CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "calibration.json")

MIN_SAMPLE_GLOBAL = 150     # minimo pronostici chiusi per la curva globale
MIN_SAMPLE_MARKET = 100     # minimo per una curva dedicata a un mercato
MIN_SAMPLE_WEIGHT = 150     # minimo per stimare il peso mercato ottimale
MIN_PER_BIN = 8             # minimo esiti in una fascia perché faccia testo
MAX_CORRECTION = 25.0       # punti percentuali: oltre = dati sporchi

_cache = {"data": None, "mtime": None}


# ============================================================
# REGRESSIONE ISOTONICA (Pool Adjacent Violators)
# ============================================================

def _pav(x: np.ndarray, y: np.ndarray, w: np.ndarray = None):
    """
    Pool Adjacent Violators: fit monotòno non decrescente di y su x.
    Ritorna (x_blocco, y_isotonico) da usare con interpolazione lineare.

    L'ancoraggio in x di ogni blocco è la media PESATA delle x fuse: usare
    il minimo del blocco (come faceva la v4) sposta la curva a sinistra e
    introduce un bias sistematico nell'interpolazione.
    """
    order = np.argsort(x)
    x, y = np.asarray(x)[order].astype(float), np.asarray(y)[order].astype(float)
    w = np.ones_like(y) if w is None else np.asarray(w)[order].astype(float)

    vals = list(y)
    wts = list(w)
    xs = list(x)
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-12:
            new_w = wts[i] + wts[i + 1]
            new_v = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / new_w
            new_x = (xs[i] * wts[i] + xs[i + 1] * wts[i + 1]) / new_w
            vals[i], wts[i], xs[i] = new_v, new_w, new_x
            del vals[i + 1], wts[i + 1], xs[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.array(xs), np.array(vals)


def _fit_curve(probs: np.ndarray, wins: np.ndarray) -> dict:
    """
    Fitta la curva di calibrazione: raggruppa per fasce di probabilità
    (riduce il rumore), poi applica PAV sulle medie di fascia.
    probs in [0,100], wins in {0,1}.

    v5: le fasce partono da 0 e non da 40, perché ora il database contiene
    anche i mercati non consigliati (probabilità basse).
    """
    probs = np.asarray(probs, dtype=float)
    wins = np.asarray(wins, dtype=int)
    ok = np.isfinite(probs)
    probs, wins = probs[ok], wins[ok]
    if probs.size == 0:
        return {}

    bins = np.arange(0, 101, 5)
    idx = np.digitize(probs, bins)
    xs, ys, ws = [], [], []
    for b in np.unique(idx):
        mask = idx == b
        if mask.sum() >= MIN_PER_BIN:
            xs.append(float(probs[mask].mean()))
            ys.append(float(wins[mask].mean() * 100))
            ws.append(int(mask.sum()))
    if len(xs) < 3:
        return {}
    px, py = _pav(np.array(xs), np.array(ys), np.array(ws))
    return {"x": [round(v, 2) for v in px.tolist()],
            "y": [round(v, 2) for v in py.tolist()],
            "n_bins": len(xs), "n": int(probs.size)}


# ============================================================
# FIT + PERSISTENZA
# ============================================================

def fit_and_save() -> dict:
    """
    Stima le curve di calibrazione (globale + per mercato dove il campione
    basta) e il peso mercato ottimale, e salva tutto in calibration.json.

    Returns: dict riepilogo dello stato.
    """
    import storage
    df = storage.all_predictions_df()
    settled = df[df["status"].isin(["won", "lost"])].copy()

    # v5: il fit gira su prob_raw (ancorata, NON calibrata). Se manca si
    # ripiega su prob solo per le righe vecchie migrate, dove coincidono.
    if "prob_raw" not in settled.columns:
        settled["prob_raw"] = settled["prob"]
    settled["prob_raw"] = settled["prob_raw"].fillna(settled["prob"])
    settled = settled[settled["prob_raw"].notna()]

    n = len(settled)
    result = {"fitted_at": datetime.now().isoformat(timespec="seconds"),
              "n_settled": n, "global": {}, "per_market": {},
              "market_weight": None, "market_weight_n": 0,
              "fit_column": "prob_raw",
              "active": False}

    result["global_pure"] = {}
    result["per_market_pure"] = {}

    if n >= MIN_SAMPLE_GLOBAL:
        wins = (settled["status"] == "won").to_numpy(dtype=int)

        # --- Curva su prob_raw: corregge ciò che l'app MOSTRA ---
        curve = _fit_curve(settled["prob_raw"].to_numpy(dtype=float), wins)
        if curve:
            result["global"] = curve
            result["active"] = True

        for mkt, grp in settled.groupby("market"):
            if len(grp) >= MIN_SAMPLE_MARKET:
                c = _fit_curve(grp["prob_raw"].to_numpy(dtype=float),
                               (grp["status"] == "won").to_numpy(dtype=int))
                if c:
                    result["per_market"][mkt] = c

        # --- Curva su prob_pure: corregge ciò che entra nell'EV ---
        # Servono DUE curve separate. prob_raw è ancorata al mercato,
        # prob_pure no: hanno distribuzioni diverse e quindi errori di
        # calibrazione diversi. Applicare la curva di prob_raw a prob_pure
        # (o viceversa) è un errore silenzioso ma sistematico.
        if "prob_pure" in settled.columns:
            pure = settled[settled["prob_pure"].notna()]
            if len(pure) >= MIN_SAMPLE_GLOBAL:
                cp = _fit_curve(pure["prob_pure"].to_numpy(dtype=float),
                                (pure["status"] == "won").to_numpy(dtype=int))
                if cp:
                    result["global_pure"] = cp
                for mkt, grp in pure.groupby("market"):
                    if len(grp) >= MIN_SAMPLE_MARKET:
                        c = _fit_curve(
                            grp["prob_pure"].to_numpy(dtype=float),
                            (grp["status"] == "won").to_numpy(dtype=int))
                        if c:
                            result["per_market_pure"][mkt] = c

    # --- Peso mercato ottimale (grid search sul Brier) ---
    if {"prob_pure", "prob_market"}.issubset(settled.columns):
        wdf = settled[settled["prob_pure"].notna() &
                      settled["prob_market"].notna()]
    else:
        wdf = settled.iloc[0:0]

    if len(wdf) >= MIN_SAMPLE_WEIGHT:
        p_model = wdf["prob_pure"].to_numpy(dtype=float) / 100.0
        p_mkt = wdf["prob_market"].to_numpy(dtype=float) / 100.0
        y = (wdf["status"] == "won").to_numpy(dtype=int)
        best_w, best_brier = None, np.inf
        for w in np.arange(0.0, 1.01, 0.05):
            blend = (1 - w) * p_model + w * p_mkt
            brier = float(np.mean((blend - y) ** 2))
            if brier < best_brier:
                best_brier, best_w = brier, float(w)
        result["market_weight"] = round(best_w, 2)
        result["market_weight_n"] = int(len(wdf))
        result["market_weight_brier"] = round(best_brier, 4)
        # Riferimenti utili: quanto valgono gli estremi della griglia
        result["brier_model_only"] = round(float(np.mean((p_model - y) ** 2)), 4)
        result["brier_market_only"] = round(float(np.mean((p_mkt - y) ** 2)), 4)

    with open(CAL_PATH, "w") as f:
        json.dump(result, f, indent=2)
    _cache["data"] = None  # invalida la cache
    return result


def load() -> Optional[dict]:
    """Carica calibration.json (con cache su mtime)."""
    if not os.path.exists(CAL_PATH):
        return None
    mtime = os.path.getmtime(CAL_PATH)
    if _cache["data"] is not None and _cache["mtime"] == mtime:
        return _cache["data"]
    try:
        with open(CAL_PATH) as f:
            data = json.load(f)
        _cache["data"], _cache["mtime"] = data, mtime
        return data
    except Exception:
        return None


def apply(prob: float, market: str = None, kind: str = "raw") -> float:
    """
    Applica la calibrazione a una probabilità NON calibrata (in %, 0-100).

    kind="raw"  → probabilità ancorata al mercato (quella che si mostra)
    kind="pure" → probabilità pura del modello (quella che entra nell'EV)

    Le due curve sono stimate separatamente perché le due grandezze hanno
    distribuzioni diverse: usare la curva sbagliata introduce un errore
    sistematico invisibile.

    ATTENZIONE: l'input deve essere la probabilità non calibrata. Passare qui
    un valore già calibrato produce una doppia correzione. L'app salva
    entrambe le versioni proprio per non confonderle (vedi storage.py).

    Se non c'è una calibrazione attiva, ritorna la probabilità invariata.
    """
    cal = load()
    if not cal or not cal.get("active"):
        return prob

    per_key = "per_market_pure" if kind == "pure" else "per_market"
    glob_key = "global_pure" if kind == "pure" else "global"

    curve = None
    if market and market in (cal.get(per_key) or {}):
        curve = cal[per_key][market]
    elif cal.get(glob_key):
        curve = cal[glob_key]
    elif kind == "pure" and cal.get("global"):
        # Nessuna curva pura ancora disponibile: meglio non correggere che
        # correggere con la curva sbagliata.
        return prob
    if not curve or len(curve.get("x", [])) < 2:
        return prob

    x = np.array(curve["x"], dtype=float)
    y = np.array(curve["y"], dtype=float)
    # Interpolazione lineare; oltre i bordi, estende l'ultimo valore
    calibrated = float(np.interp(prob, x, y))
    # Non permettere correzioni assurde: segnale di dati sporchi
    return float(np.clip(calibrated, prob - MAX_CORRECTION,
                         prob + MAX_CORRECTION))


def is_active() -> bool:
    cal = load()
    return bool(cal and cal.get("active"))


def status() -> dict:
    """Stato leggibile per la pagina Performance."""
    cal = load()
    if not cal:
        return {"active": False, "n_settled": 0, "fitted_at": None,
                "per_market": [], "market_weight": None,
                "market_weight_n": 0, "min_required": MIN_SAMPLE_GLOBAL,
                "brier_model_only": None, "brier_market_only": None}
    return {"active": cal.get("active", False),
            "n_settled": cal.get("n_settled", 0),
            "fitted_at": cal.get("fitted_at"),
            "per_market": sorted(cal.get("per_market", {}).keys()),
            "market_weight": cal.get("market_weight"),
            "market_weight_n": cal.get("market_weight_n", 0),
            "min_required": MIN_SAMPLE_GLOBAL,
            "pure_curve": bool(cal.get("global_pure")),
            "brier_model_only": cal.get("brier_model_only"),
            "brier_market_only": cal.get("brier_market_only")}
