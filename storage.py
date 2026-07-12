"""
💾 STORAGE — Persistenza pronostici su SQLite
==============================================
Sostituisce il tracker in session_state (che moriva alla chiusura del browser)
e il vecchio tracker.py su Excel. Database locale: betengine.db nella cartella
del progetto. Nessuna dipendenza nuova: sqlite3 è nella standard library.

Schema tabella `predictions`:
- Identità partita: fixture_id, match_date, league_id, league, home, away
- Pronostico: market (1X2/OU/BTTS/Cards), selection (label), selection_code
  (codice macchina: 1, X, 2, O2.5, U2.5, GG, NG, O3.5cards, ...)
- Probabilità: prob (ancorata, mostrata), prob_pure (per EV), odds, ev_pct
- Esito: status (pending/won/lost/void), score_home, score_away, total_cards

Il vincolo UNIQUE(fixture_id, selection_code) impedisce i duplicati:
salvare due volte la stessa analisi non sporca il database.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "betengine.db")


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Crea la tabella se non esiste. Chiamata all'avvio dell'app."""
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at     TEXT NOT NULL,
                match_date     TEXT,
                fixture_id     INTEGER,
                league_id      INTEGER,
                league         TEXT,
                home           TEXT,
                away           TEXT,
                market         TEXT,
                selection      TEXT,
                selection_code TEXT,
                prob           REAL,
                prob_pure      REAL,
                odds           REAL,
                ev_pct         REAL,
                stars          INTEGER,
                anchored       INTEGER DEFAULT 0,
                status         TEXT DEFAULT 'pending',
                score_home     INTEGER,
                score_away     INTEGER,
                total_cards    INTEGER,
                settled_at     TEXT,
                UNIQUE(fixture_id, selection_code)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_status ON predictions(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_fixture ON predictions(fixture_id)")


def save_predictions(rows: list) -> int:
    """
    Salva una lista di pronostici. I duplicati (stessa fixture + stessa
    selezione) vengono ignorati silenziosamente grazie al vincolo UNIQUE.

    Returns: numero di righe effettivamente inserite.
    """
    if not rows:
        return 0
    inserted = 0
    now = datetime.now().isoformat(timespec="seconds")
    with _conn() as c:
        for r in rows:
            cur = c.execute("""
                INSERT OR IGNORE INTO predictions
                (created_at, match_date, fixture_id, league_id, league,
                 home, away, market, selection, selection_code,
                 prob, prob_pure, odds, ev_pct, stars, anchored)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                now, r.get("match_date"), r.get("fixture_id"),
                r.get("league_id"), r.get("league"),
                r.get("home"), r.get("away"), r.get("market"),
                r.get("selection"), r.get("selection_code"),
                r.get("prob"), r.get("prob_pure"), r.get("odds"),
                r.get("ev_pct"), r.get("stars"),
                1 if r.get("anchored") else 0,
            ))
            inserted += cur.rowcount
    return inserted


def fixture_saved_selections(fixture_id: int) -> set:
    """Codici selezione già salvati per una fixture (per il check ✓ nell'app)."""
    with _conn() as c:
        cur = c.execute(
            "SELECT selection_code FROM predictions WHERE fixture_id = ?",
            (fixture_id,))
        return {row["selection_code"] for row in cur.fetchall()}


def pending_predictions() -> list:
    """Pronostici in attesa di esito, come lista di dict."""
    with _conn() as c:
        cur = c.execute("""
            SELECT * FROM predictions
            WHERE status = 'pending'
            ORDER BY match_date ASC
        """)
        return [dict(row) for row in cur.fetchall()]


def settle_prediction(pred_id: int, status: str,
                      score_home=None, score_away=None, total_cards=None):
    """Registra l'esito di un pronostico (won/lost/void)."""
    with _conn() as c:
        c.execute("""
            UPDATE predictions
            SET status = ?, score_home = ?, score_away = ?,
                total_cards = ?, settled_at = ?
            WHERE id = ?
        """, (status, score_home, score_away, total_cards,
              datetime.now().isoformat(timespec="seconds"), pred_id))


def all_predictions_df() -> pd.DataFrame:
    """Tutto il database come DataFrame (per export e pagina Performance)."""
    with _conn() as c:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY match_date DESC, id DESC", c)


def counts() -> dict:
    """Conteggi rapidi per la sidebar."""
    with _conn() as c:
        cur = c.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN status IN ('won','lost') THEN 1 ELSE 0 END) AS settled
            FROM predictions
        """)
        row = cur.fetchone()
        return {"total": row["total"] or 0,
                "pending": row["pending"] or 0,
                "settled": row["settled"] or 0}


def delete_all() -> int:
    """Svuota il database. Ritorna il numero di righe eliminate."""
    with _conn() as c:
        cur = c.execute("DELETE FROM predictions")
        return cur.rowcount


init_db()
