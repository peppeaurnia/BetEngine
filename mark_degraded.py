#!/usr/bin/env python3
"""
🏷️  MARK DEGRADED — marca a posteriori i pronostici prodotti con dati poveri

PERCHÉ SERVE
Le previsioni fatte a inizio stagione nascono da statistiche di squadra quasi
vuote (due o tre partite giocate): non sono previsioni sbagliate in modo
correggibile, sono previsioni senza informazione. Se finiscono nel fit della
calibrazione insieme a quelle di dicembre, la curva impara una relazione media
tra due regimi diversi e non descrive né l'uno né l'altro.

Da v5.2 l'app marca automaticamente queste righe (`degraded = 1`). Questo
script serve solo per le righe salvate PRIMA dell'aggiornamento.

ATTENZIONE — FUNZIONA UNA VOLTA SOLA
Marcare per data è un'approssimazione grossolana: presuppone che TUTTO ciò che
sta nel database prima di una certa data sia degenere. È vero adesso, perché
hai iniziato a salvare a stagione appena cominciata. Non sarà più vero quando
il database conterrà mesi diversi mescolati, e allora nessun filtro per data
potrà distinguerli. È esattamente il motivo per cui il flag va messo prima di
continuare a salvare.

USO (dalla cartella del progetto, quella con betengine.db dentro):

    python mark_degraded.py                  # anteprima, non scrive nulla
    python mark_degraded.py --fino 2026-09-30 --applica

    python mark_degraded.py --annulla        # rimette degraded = 0 ovunque
"""

import argparse
import os
import sqlite3
import sys

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "betengine.db")


def _connect():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database non trovato: {DB_PATH}")
        print("   Lancia questo script dalla cartella del progetto, quella")
        print("   che contiene betengine.db.")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _check_schema(conn):
    cols = {r["name"] for r in
            conn.execute("PRAGMA table_info(predictions)").fetchall()}
    if "degraded" not in cols:
        print("❌ La colonna `degraded` non esiste ancora.")
        print("   Sostituisci prima storage.py con la versione v5.2 e avvia")
        print("   l'app una volta: la migrazione crea la colonna da sola.")
        sys.exit(1)


def preview(conn, fino):
    print("=" * 60)
    print("SITUAZIONE ATTUALE")
    print("=" * 60)

    tot = conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    deg = conn.execute("SELECT COUNT(*) c FROM predictions "
                       "WHERE degraded = 1").fetchone()["c"]
    print(f"   Righe totali: {tot}")
    print(f"   Già marcate degraded: {deg}")

    rows = conn.execute(
        "SELECT match_date, COUNT(*) n, "
        "       SUM(CASE WHEN odds IS NOT NULL THEN 1 ELSE 0 END) con_quota, "
        "       ROUND(AVG(mu_total), 2) mu "
        "FROM predictions GROUP BY match_date ORDER BY match_date"
    ).fetchall()

    if rows:
        print("\n   Per data (mu = gol attesi medi; sotto 2.0 = dati degeneri):")
        print(f"   {'data':<14}{'righe':>7}{'con quota':>11}{'mu medio':>10}")
        for r in rows[:25]:
            mu = f"{r['mu']:.2f}" if r["mu"] is not None else "n/d"
            print(f"   {str(r['match_date'] or '?'):<14}{r['n']:>7}"
                  f"{r['con_quota']:>11}{mu:>10}")
        if len(rows) > 25:
            print(f"   … e altre {len(rows) - 25} date")

    target = conn.execute(
        "SELECT COUNT(*) c FROM predictions "
        "WHERE match_date <= ? AND (degraded IS NULL OR degraded = 0)",
        (fino,)).fetchone()["c"]

    print("\n" + "=" * 60)
    print(f"AZIONE PROPOSTA: marcare degraded = 1 su tutto ciò che ha")
    print(f"match_date <= {fino}")
    print("=" * 60)
    print(f"   Righe che verrebbero marcate: {target}")
    print(f"   Righe che resterebbero buone: {tot - deg - target}")
    return target


def apply(conn, fino):
    cur = conn.execute(
        "UPDATE predictions SET degraded = 1 "
        "WHERE match_date <= ? AND (degraded IS NULL OR degraded = 0)",
        (fino,))
    conn.commit()
    return cur.rowcount


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--fino", default="2026-09-30",
                    help="data limite inclusa, formato AAAA-MM-GG "
                         "(default: 2026-09-30)")
    ap.add_argument("--applica", action="store_true",
                    help="esegue davvero la modifica")
    ap.add_argument("--annulla", action="store_true",
                    help="rimette degraded = 0 su tutte le righe")
    args = ap.parse_args()

    conn = _connect()
    _check_schema(conn)

    if args.annulla:
        n = conn.execute("UPDATE predictions SET degraded = 0").rowcount
        conn.commit()
        print(f"✅ Flag rimosso da {n} righe.")
        return

    n = preview(conn, args.fino)

    if not args.applica:
        print("\n   Anteprima soltanto: non è stato modificato nulla.")
        print(f"   Per eseguire: python mark_degraded.py "
              f"--fino {args.fino} --applica")
        return

    if n == 0:
        print("\n   Nessuna riga da marcare.")
        return

    print(f"\n   Fai una copia di betengine.db prima di procedere.")
    risposta = input(f"   Marcare {n} righe come degradate? [s/N] ").strip().lower()
    if risposta not in ("s", "si", "sì", "y", "yes"):
        print("   Annullato, nessuna modifica.")
        return

    done = apply(conn, args.fino)
    print(f"\n✅ {done} righe marcate come degradate.")
    print("   Restano nel database e le vedi ancora nella pagina Performance,")
    print("   ma sono escluse dal fit della calibrazione.")
    print("   Da adesso in poi il flag viene messo automaticamente dall'app.")


if __name__ == "__main__":
    main()
