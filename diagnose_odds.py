#!/usr/bin/env python3
"""
🔍 DIAGNOSE ODDS — perché non arrivano le quote?

Senza quote BetEngine non può calcolare l'EV, quindi non può fare il suo
lavoro: ripiega sulle soglie di probabilità e diventa un generatore di
opinioni. Questo script isola la causa, invece di farti indovinare.

USO:
    python diagnose_odds.py              # controlla tutte le leghe mappate
    python diagnose_odds.py 135          # solo Serie A

Costo: 1 chiamata a The Odds API per lega controllata, più 1 per il
controllo della quota residua (quest'ultima è gratuita).
"""

import sys
import requests

from odds_api import LEAGUE_MAPPING, SHARP_BOOKS, get_odds_api_key, \
    _odds_by_bookmaker, _pick_sharp_book, _book_margin

BASE = "https://api.the-odds-api.com/v4"


def check_key(api_key):
    """Chiave valida? Quota residua?"""
    print("=" * 62)
    print("1. CHIAVE E QUOTA")
    print("=" * 62)
    try:
        r = requests.get(f"{BASE}/sports", params={"apiKey": api_key}, timeout=15)
    except Exception as e:
        print(f"   ❌ Rete non raggiungibile: {e}")
        return False

    if r.status_code == 401:
        print("   ❌ Chiave rifiutata (401). Verifica THE_ODDS_API_KEY.")
        return False
    if r.status_code == 429:
        print("   ❌ Quota esaurita (429). È la causa più comune se prima "
              "funzionava.")
        return False
    if r.status_code != 200:
        print(f"   ❌ Risposta inattesa: HTTP {r.status_code}")
        return False

    used = r.headers.get("x-requests-used", "?")
    left = r.headers.get("x-requests-remaining", "?")
    print(f"   ✅ Chiave valida · richieste usate: {used} · rimanenti: {left}")
    try:
        if int(left) < 20:
            print("   ⚠️  Quota quasi finita: le prossime analisi "
                  "resteranno senza quote.")
    except (TypeError, ValueError):
        pass

    active = {s["key"] for s in r.json() if s.get("active")}
    print(f"   Sport/leghe attivi in questo momento: {len(active)}")
    return active


def check_league(api_key, league_id, active_keys):
    sport = LEAGUE_MAPPING.get(league_id)
    print("-" * 62)
    print(f"Lega {league_id} → {sport}")

    if not sport:
        print("   ❌ Non mappata in odds_api.LEAGUE_MAPPING. Aggiungila.")
        return

    if active_keys and sport not in active_keys:
        print("   ⚠️  Il campionato NON è attivo su The Odds API in questo "
              "momento.")
        print("      Fuori stagione, o pausa: i bookmaker non quotano ancora. "
              "Nessun bug, semplicemente non c'è mercato.")
        return

    try:
        r = requests.get(f"{BASE}/sports/{sport}/odds",
                         params={"apiKey": api_key, "regions": "eu,uk",
                                 "markets": "h2h,totals,btts",
                                 "oddsFormat": "decimal"}, timeout=20)
    except Exception as e:
        print(f"   ❌ Errore di rete: {e}")
        return

    if r.status_code != 200:
        print(f"   ❌ HTTP {r.status_code}: {r.text[:160]}")
        return

    events = r.json()
    if not events:
        print("   ⚠️  Nessun evento quotato. Le partite sono troppo lontane "
              "(i book aprono di norma 3-7 giorni prima) o già iniziate.")
        return

    print(f"   ✅ {len(events)} partite quotate")
    ev = events[0]
    print(f"   Esempio: {ev.get('home_team')} vs {ev.get('away_team')} "
          f"({ev.get('commence_time', '')[:16]})")

    per_book = _odds_by_bookmaker(ev)
    print(f"   Bookmaker con quote utilizzabili: {len(per_book)}")

    present = [b for b in SHARP_BOOKS if b in per_book]
    if present:
        print(f"   ✅ Book sharp disponibili: {', '.join(present)}")
    else:
        print("   ⚠️  Nessun book sharp (Pinnacle/exchange) per questa lega.")
        print("      Si ripiega sul book col margine più basso: le "
              "probabilità di mercato saranno meno affidabili.")

    key, sharp = _pick_sharp_book(per_book)
    if key:
        m = _book_margin(sharp)
        print(f"   Book scelto per le probabilità: {key} "
              f"(margine medio {m:.4f} → {(m - 1) * 100:.2f}%)")
        print(f"   Quote: " + " · ".join(f"{k} {v}" for k, v in
                                         sorted(sharp.items())[:7]))
    else:
        print("   ❌ Nessun book quota un gruppo COMPLETO di esiti. "
              "Impossibile rimuovere il margine.")

    # I nomi squadra sono la causa silenziosa più frequente
    print("\n   Nomi squadra su The Odds API (per confronto con API-Football):")
    for e in events[:5]:
        print(f"     · {e.get('home_team')}  vs  {e.get('away_team')}")
    print("   Se questi non somigliano ai nomi che vedi nell'app, il "
          "matching fuzzy fallisce e le quote non vengono associate.")


def main():
    api_key = get_odds_api_key()
    if not api_key:
        print("❌ Nessuna chiave trovata. Impostala in config.py "
              "(THE_ODDS_API_KEY) o in st.secrets.")
        sys.exit(1)

    active = check_key(api_key)
    if active is False:
        sys.exit(1)

    print()
    print("=" * 62)
    print("2. LEGHE")
    print("=" * 62)

    leagues = [int(sys.argv[1])] if len(sys.argv) > 1 else list(LEAGUE_MAPPING)
    for lid in leagues:
        check_league(api_key, lid, active)

    print("-" * 62)
    print("\nRiepilogo delle cause tipiche, in ordine di frequenza:")
    print("  1. Campionato non ancora attivo (fuori stagione / pausa)")
    print("  2. Partita troppo lontana: i book aprono 3-7 giorni prima")
    print("  3. Quota mensile di The Odds API esaurita")
    print("  4. Nomi squadra che non combaciano tra le due API")
    print("  5. Lega assente da odds_api.LEAGUE_MAPPING")


if __name__ == "__main__":
    main()
