"""
🎯 CALIBRATION — Calibrazione empirica data-driven
====================================================
Sostituisce le correzioni fatte a mano su campioni piccoli (dead zone
65-70%, soglie ritoccate a orecchio) con una calibrazione isotonica
stimata sui pronostici REALMENTE chiusi nel database (betengine.db).

Come funziona:
1. Quando il campione di pronostici chiusi è sufficiente (>= 150),
   `fit_and_save()` stima la relazione monotòna tra probabilità dichiarata
   e frequenza reale (regressione isotonica via Pool Adjacent Violators,
   implementata a mano: nessuna dipendenza da sklearn).
2. La curva viene salvata in calibration.json.
3. L'app, tramite `apply()`, corregge le probabilità mostrate:
   se il modello storicamente dice 68% ma vince il 61%, un nuovo 68%
   diventa 61%. Le soglie dei pronostici si applicano al valore corretto.
4. `estimate_market_weight()` stima inoltre il peso di anchoring ottimale
   (quello che minimizza il Brier score sui dati reali), da confrontare
   con il MARKET_WEIGHT=0.35 attuale calibrato su un solo mese.

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

_cache = {"data": None, "mtime": None}


# ============================================================
# REGRESSIONE ISOTONICA (Pool Adjacent Violators)
# ============================================================

def _pav(x: np.ndarray, y: np.ndarray, w: np.ndarray = None):
    """
    Pool Adjacent Violators: fit monotòno non decrescente di y su x.
    Ritorna (x_ordinato, y_isotonico) da usare con interpolazione lineare.
    """
    order = np.argsort(x)
    x, y = x[order], y[order].astype(float)
    w = np.ones_like(y) if w is None else w[order].astype(float)

    # Blocchi (valore, peso); fondi i blocchi che violano la monotonia
    vals = list(y)
    wts = list(w)
    xs = list(x)
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-12:
            new_w = wts[i] + wts[i + 1]
            new_v = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / new_w
            vals[i] = new_v
            wts[i] = new_w
            xs[i] = (xs[i] * wts[i - 1] if False else xs[i])  # x resta il min del blocco
            del vals[i + 1], wts[i + 1], xs[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.array(xs), np.array(vals)


def _fit_curve(probs: np.ndarray, wins: np.ndarray) -> dict:
    """
    Fitta la curva di calibrazione: raggruppa per fasce di probabilità
    (per ridurre il rumore), poi applica PAV sulle medie di fascia.
    probs in [0,100], wins in {0,1}.
    """
    bins = np.arange(40, 101, 5)
    idx = np.digitize(probs, bins)
    xs, ys, ws = [], [], []
    for b in np.unique(idx):
        mask = idx == b
        if mask.sum() >= 5:
            xs.append(float(probs[mask].mean()))
            ys.append(float(wins[mask].mean() * 100))
            ws.append(int(mask.sum()))
    if len(xs) < 3:
        return {}
    px, py = _pav(np.array(xs), np.array(ys), np.array(ws))
    return {"x": [round(v, 2) for v in px.tolist()],
            "y": [round(v, 2) for v in py.tolist()]}


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
    n = len(settled)

    result = {"fitted_at": datetime.now().isoformat(timespec="seconds"),
              "n_settled": n, "global": {}, "per_market": {},
              "market_weight": None, "market_weight_n": 0,
              "active": False}

    if n >= MIN_SAMPLE_GLOBAL:
        probs = settled["prob"].to_numpy(dtype=float)
        wins = (settled["status"] == "won").to_numpy(dtype=int)
        curve = _fit_curve(probs, wins)
        if curve:
            result["global"] = curve
            result["active"] = True

        for mkt, grp in settled.groupby("market"):
            if len(grp) >= MIN_SAMPLE_MARKET:
                c = _fit_curve(grp["prob"].to_numpy(dtype=float),
                               (grp["status"] == "won").to_numpy(dtype=int))
                if c:
                    result["per_market"][mkt] = c

    # --- Peso mercato ottimale (grid search sul Brier) ---
    wdf = settled[settled["prob_pure"].notna() &
                  settled["prob_market"].notna()] \
        if "prob_market" in settled.columns else settled.iloc[0:0]
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


def apply(prob: float, market: str = None) -> float:
    """
    Applica la calibrazione a una probabilità (in %, 0-100).
    Se non c'è una calibrazione attiva, ritorna la probabilità invariata.
    Usa la curva del mercato specifico se disponibile, altrimenti la globale.
    """
    cal = load()
    if not cal or not cal.get("active"):
        return prob

    curve = None
    if market and market in cal.get("per_market", {}):
        curve = cal["per_market"][market]
    elif cal.get("global"):
        curve = cal["global"]
    if not curve or len(curve.get("x", [])) < 2:
        return prob

    x = np.array(curve["x"], dtype=float)
    y = np.array(curve["y"], dtype=float)
    # Interpolazione lineare; oltre i bordi, estende l'ultimo valore
    calibrated = float(np.interp(prob, x, y))
    # Non permettere correzioni assurde (> 25 punti): segnale di dati sporchi
    return float(np.clip(calibrated, prob - 25, prob + 25))


def is_active() -> bool:
    cal = load()
    return bool(cal and cal.get("active"))


def status() -> dict:
    """Stato leggibile per la pagina Performance."""
    cal = load()
    if not cal:
        return {"active": False, "n_settled": 0, "fitted_at": None,
                "per_market": [], "market_weight": None,
                "market_weight_n": 0, "min_required": MIN_SAMPLE_GLOBAL}
    return {"active": cal.get("active", False),
            "n_settled": cal.get("n_settled", 0),
            "fitted_at": cal.get("fitted_at"),
            "per_market": sorted(cal.get("per_market", {}).keys()),
            "market_weight": cal.get("market_weight"),
            "market_weight_n": cal.get("market_weight_n", 0),
            "min_required": MIN_SAMPLE_GLOBAL}
