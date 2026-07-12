"""
⚙️ DC MODEL — Dixon-Coles con fit MLE e decadimento temporale
===============================================================
Il salto di qualità del modello (punto 5 della roadmap): invece di derivare
i μ dalle medie stagionali aggregate di API-Football, stima attacco e difesa
di ogni squadra con un fit di massima verosimiglianza PARTITA PER PARTITA
sulle ultime 2 stagioni, pesando le partite recenti più di quelle vecchie
(decadimento esponenziale, Dixon & Coles 1997).

Vantaggi rispetto alle medie aggregate:
- Le partite di 8 mesi fa pesano meno di quelle di 2 settimane fa
  (il decadimento sostituisce in modo principiato form_factor e momentum)
- Tiene conto della forza degli avversari incontrati
- ρ (correzione punteggi bassi) stimato dai dati invece che fissato

USO (da terminale, NON dall'app — il fit costa chiamate API e CPU):
    python dc_model.py fit 135          # fitta la Serie A
    python dc_model.py fit-all          # fitta tutte le leghe (≈2 chiamate/lega)
    python dc_model.py show 135         # mostra i rating della Serie A

I parametri vengono salvati in dc_params.json. L'app li carica se esistono
e non sono più vecchi di MAX_AGE_DAYS: in quel caso i mercati GOL vengono
calcolati con Dixon-Coles (i cartellini restano al modello v4) e il
pronostico viene tracciato con engine='dc' per il confronto A/B nella
pagina Performance. Se i parametri mancano o sono stantii → fallback
automatico al motore v4. Rifitta 1-2 volte a settimana.

Costo API: 1 chiamata per lega-stagione-pagina (~2-3 per lega).
Richiede: scipy (aggiunto a requirements.txt).
"""

import json
import os
import sys
from datetime import datetime, date
from typing import Dict, Optional

import numpy as np
import requests
from scipy.optimize import minimize
from scipy.stats import poisson

from probability_engine import (calculate_1x2, calculate_over_under,
                                calculate_btts, calculate_exact_scores,
                                MAX_GOALS)

BASE_URL = "https://v3.football.api-sports.io"
PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "dc_params.json")

XI = 0.0019          # decadimento/giorno (Dixon-Coles ~0.0065 per mezza settimana)
                     # → una partita di 1 anno fa pesa exp(-0.0019*365) ≈ 0.50
MAX_AGE_DAYS = 10    # oltre questa età i parametri sono stantii → fallback v4
MIN_MATCHES_TEAM = 8 # sotto questo numero di partite una squadra non è affidabile

LEAGUE_IDS = [135, 39, 140, 78, 61, 94, 88]  # Serie A, PL, Liga, Bundes, L1, Primeira, Eredivisie


# ============================================================
# FETCH PARTITE STORICHE
# ============================================================

def _headers(api_key):
    return {"x-apisports-key": api_key}


def fetch_league_matches(api_key: str, league_id: int, seasons: list) -> list:
    """
    Scarica tutte le partite CONCLUSE di una lega per le stagioni date.
    ~1 chiamata API per stagione (più eventuali pagine).

    Returns: lista di dict {date, home_id, home, away_id, away, hg, ag}
    """
    matches = []
    for season in seasons:
        page = 1
        while True:
            r = requests.get(f"{BASE_URL}/fixtures",
                             params={"league": league_id, "season": season,
                                     "status": "FT-AET-PEN", "page": page},
                             headers=_headers(api_key), timeout=20)
            r.raise_for_status()
            data = r.json()
            for fx in data.get("response", []):
                goals = fx.get("goals", {})
                if goals.get("home") is None or goals.get("away") is None:
                    continue
                matches.append({
                    "date": fx["fixture"]["date"][:10],
                    "home_id": fx["teams"]["home"]["id"],
                    "home": fx["teams"]["home"]["name"],
                    "away_id": fx["teams"]["away"]["id"],
                    "away": fx["teams"]["away"]["name"],
                    "hg": int(goals["home"]),
                    "ag": int(goals["away"]),
                })
            paging = data.get("paging", {})
            if paging.get("current", 1) >= paging.get("total", 1):
                break
            page += 1
    return matches


# ============================================================
# FIT DIXON-COLES (MLE con decadimento temporale)
# ============================================================

def _tau(x, y, lam, mu, rho):
    """Correzione Dixon-Coles per i punteggi bassi (dipendenza 0-0/1-1)."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def fit_dixon_coles(matches: list, xi: float = XI) -> Optional[dict]:
    """
    Massima verosimiglianza pesata:
      L = Σ w_i · [log τ + log Pois(hg; λ_i) + log Pois(ag; μ_i)]
      λ_i = exp(att_h + def_a + γ)   μ_i = exp(att_a + def_h)
      w_i = exp(-ξ · giorni_trascorsi)

    Identificabilità: media degli attacchi vincolata a 0 (penalità).
    Returns: dict parametri o None se il fit fallisce.
    """
    if not matches:
        return None

    teams = sorted({m["home_id"] for m in matches} |
                   {m["away_id"] for m in matches})
    t_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    names = {}
    counts = {t: 0 for t in teams}

    today = date.today()
    rows = []
    for m in matches:
        d = datetime.strptime(m["date"], "%Y-%m-%d").date()
        days = (today - d).days
        w = float(np.exp(-xi * max(days, 0)))
        rows.append((t_idx[m["home_id"]], t_idx[m["away_id"]],
                     m["hg"], m["ag"], w))
        names[m["home_id"]] = m["home"]
        names[m["away_id"]] = m["away"]
        counts[m["home_id"]] += 1
        counts[m["away_id"]] += 1

    hi = np.array([r[0] for r in rows])
    ai = np.array([r[1] for r in rows])
    hg = np.array([r[2] for r in rows])
    ag = np.array([r[3] for r in rows])
    w = np.array([r[4] for r in rows])

    def neg_ll(params):
        att = params[:n]
        dfc = params[n:2 * n]
        gamma, rho = params[2 * n], params[2 * n + 1]

        lam = np.exp(att[hi] + dfc[ai] + gamma)
        mu = np.exp(att[ai] + dfc[hi])
        lam = np.clip(lam, 1e-6, 8.0)
        mu = np.clip(mu, 1e-6, 8.0)

        ll = (poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu))

        # Correzione τ solo dove serve (punteggi 0/1)
        tau = np.ones_like(lam)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        tau = np.clip(tau, 1e-6, None)

        total = float(np.sum(w * (ll + np.log(tau))))
        # Penalità di identificabilità: media attacchi = 0
        total -= 1000.0 * float(np.mean(att)) ** 2
        return -total

    x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])
    bounds = ([(-2.0, 2.0)] * (2 * n)) + [(0.0, 0.8), (-0.2, 0.2)]

    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 400})
    if not res.success and res.status not in (1, 2):  # 1/2 = limiti iterazioni, ok
        print(f"⚠️ Ottimizzazione non convergente: {res.message}")

    att = res.x[:n]
    dfc = res.x[n:2 * n]
    gamma, rho = float(res.x[2 * n]), float(res.x[2 * n + 1])

    return {
        "fitted_at": datetime.now().isoformat(timespec="seconds"),
        "xi": xi,
        "home_adv": round(gamma, 4),
        "rho": round(rho, 4),
        "n_matches": len(matches),
        "neg_ll": round(float(res.fun), 2),
        "teams": {
            str(t): {
                "name": names.get(t, str(t)),
                "att": round(float(att[t_idx[t]]), 4),
                "def": round(float(dfc[t_idx[t]]), 4),
                "n": counts[t],
            } for t in teams
        },
    }


# ============================================================
# PERSISTENZA
# ============================================================

def _load_all() -> dict:
    if not os.path.exists(PARAMS_PATH):
        return {}
    try:
        with open(PARAMS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_league_params(league_id: int, params: dict):
    allp = _load_all()
    allp[str(league_id)] = params
    with open(PARAMS_PATH, "w") as f:
        json.dump(allp, f, indent=2)


def load_league_params(league_id: int,
                       max_age_days: int = MAX_AGE_DAYS) -> Optional[dict]:
    """Parametri di una lega, solo se abbastanza freschi. Altrimenti None."""
    p = _load_all().get(str(league_id))
    if not p:
        return None
    try:
        fitted = datetime.fromisoformat(p["fitted_at"]).date()
        if (date.today() - fitted).days > max_age_days:
            return None
    except Exception:
        return None
    return p


def params_age_days(league_id: int) -> Optional[int]:
    p = _load_all().get(str(league_id))
    if not p:
        return None
    try:
        return (date.today() -
                datetime.fromisoformat(p["fitted_at"]).date()).days
    except Exception:
        return None


# ============================================================
# PREDIZIONE (chiavi compatibili con probability_engine)
# ============================================================

def dc_match_probabilities(league_id: int, home_id: int,
                           away_id: int) -> Optional[dict]:
    """
    Probabilità dei mercati GOL con parametri Dixon-Coles.
    Ritorna None se: parametri assenti/stantii, squadra non nel fit,
    o squadra con troppe poche partite (neopromosse a inizio stagione).
    Il chiamante fa fallback al motore v4.

    Le chiavi ricalcano quelle di calculate_match_probabilities così
    l'output può sovrascrivere i soli mercati gol del dict v4.
    """
    p = load_league_params(league_id)
    if not p:
        return None
    th = p["teams"].get(str(home_id))
    ta = p["teams"].get(str(away_id))
    if not th or not ta:
        return None
    if th["n"] < MIN_MATCHES_TEAM or ta["n"] < MIN_MATCHES_TEAM:
        return None

    gamma, rho = p["home_adv"], p["rho"]
    mu_home = float(np.exp(th["att"] + ta["def"] + gamma))
    mu_away = float(np.exp(ta["att"] + th["def"]))
    mu_home = float(np.clip(mu_home, 0.15, 4.5))
    mu_away = float(np.clip(mu_away, 0.15, 4.5))

    # Matrice: Poisson indipendenti + correzione τ sulle 4 celle basse
    size = MAX_GOALS + 1
    ph = poisson.pmf(np.arange(size), mu_home)
    pa = poisson.pmf(np.arange(size), mu_away)
    M = np.outer(ph, pa)
    for x in (0, 1):
        for y in (0, 1):
            M[x, y] *= max(_tau(x, y, mu_home, mu_away, rho), 1e-9)
    M /= M.sum()

    probs_ou = {}
    for line in [1.5, 2.5, 3.5, 4.5]:
        ou = calculate_over_under(M, line)
        probs_ou[f"over_{line}"] = ou["p_over"]
        probs_ou[f"under_{line}"] = ou["p_under"]

    return {
        "mu_home": round(mu_home, 3),
        "mu_away": round(mu_away, 3),
        "total_expected_goals": round(mu_home + mu_away, 2),
        **calculate_1x2(M),
        **calculate_btts(M),
        **probs_ou,
        "exact_scores": calculate_exact_scores(M, top_n=10),
        "matrix": M,
        "dc_rho": rho,
        "dc_fitted_at": p["fitted_at"],
    }


# ============================================================
# CLI
# ============================================================

def _current_seasons() -> list:
    """Stagione corrente + precedente (la stagione europea parte ad agosto)."""
    today = date.today()
    y = today.year if today.month >= 8 else today.year - 1
    return [y - 1, y]


def _fit_league(api_key: str, league_id: int):
    seasons = _current_seasons()
    print(f"→ Lega {league_id}: scarico stagioni {seasons}…")
    matches = fetch_league_matches(api_key, league_id, seasons)
    print(f"  {len(matches)} partite concluse")
    if len(matches) < 100:
        print("  ⚠️ Troppo poche partite, salto il fit.")
        return
    params = fit_dixon_coles(matches)
    if params:
        save_league_params(league_id, params)
        print(f"  ✅ Fit ok: γ(casa)={params['home_adv']}, ρ={params['rho']}, "
              f"{len(params['teams'])} squadre → dc_params.json")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("fit", "fit-all", "show"):
        print(__doc__)
        return

    from config import API_FOOTBALL_KEY

    cmd = sys.argv[1]
    if cmd == "fit":
        _fit_league(API_FOOTBALL_KEY, int(sys.argv[2]))
    elif cmd == "fit-all":
        for lid in LEAGUE_IDS:
            _fit_league(API_FOOTBALL_KEY, lid)
    elif cmd == "show":
        p = _load_all().get(sys.argv[2])
        if not p:
            print("Nessun parametro per questa lega. Esegui prima il fit.")
            return
        print(f"Fit del {p['fitted_at']} su {p['n_matches']} partite · "
              f"vantaggio casa e^{p['home_adv']} = "
              f"{np.exp(p['home_adv']):.2f}x · ρ = {p['rho']}")
        teams = sorted(p["teams"].values(),
                       key=lambda t: t["att"] - t["def"], reverse=True)
        print(f"{'Squadra':<28}{'Attacco':>9}{'Difesa':>9}{'Partite':>9}")
        for t in teams:
            print(f"{t['name']:<28}{t['att']:>9.3f}{t['def']:>9.3f}{t['n']:>9}")


if __name__ == "__main__":
    main()
