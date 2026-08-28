"""
💾 STORAGE — Persistenza pronostici su SQLite
==============================================
Database locale: betengine.db nella cartella del progetto.
Nessuna dipendenza nuova: sqlite3 è nella standard library.

v5 — TRE CAMBIAMENTI STRUTTURALI

1. `prob_raw` (NUOVA COLONNA, obbligatoria per la calibrazione)
   Prima si salvava solo `prob`, che è la probabilità GIÀ calibrata.
   calibration.fit_and_save() fittava la curva su quella stessa colonna e poi
   la applicava di nuovo alle probabilità grezze: un loop di feedback che a
   ogni refit componeva la correzione su sé stessa. Ora si salvano entrambe:
     - prob_raw : ancorata al mercato, NON calibrata  → input del fit
     - prob     : quello che l'app ha effettivamente mostrato → display/audit
     - prob_pure: modello puro, senza anchoring       → EV

2. `shortlisted` (NUOVA COLONNA)
   Prima si salvavano solo i 2-3 pronostici consigliati, cioè esclusivamente
   la coda ad alta probabilità. Risultato: la curva di calibrazione poteva
   essere stimata solo su quella fascia, il Brier era misurato su un campione
   censurato, e per arrivare a 150 esiti servivano mesi.
   Ora si salva OGNI mercato di ogni partita analizzata; `shortlisted=1`
   marca quelli effettivamente consigliati. Costo su SQLite: nullo.
   Guadagno: 3-4x i dati, l'intero range di probabilità, e un confronto
   modello-vs-mercato onesto.

3. `mu_total`, `data_quality`, `degraded` (v5.2)
   Una previsione fatta alla 2ª giornata, con due partite di storico, e una
   fatta alla 25ª non sono lo stesso oggetto: la prima è quasi rumore. Senza
   marcarle, la curva di calibrazione le mescola e impara una relazione media
   che non descrive nessuno dei due regimi — e filtrare per data dopo non
   funziona, perché i campionati partono in momenti diversi e le neopromosse
   restano degeneri più a lungo. `degraded=1` marca le righe da tenere fuori
   dal fit pur conservandole nel database.

4. Colonne per il CLV (`odds_close`, `clv_pct`)
   Il Closing Line Value è il predittore più affidabile di profittabilità a
   lungo termine ed è misurabile in poche settimane, mentre lo yield richiede
   centinaia di esiti. Vedi results_updater.update_closing_odds().

Il vincolo UNIQUE(fixture_id, selection_code) impedisce i duplicati.
Le migrazioni sono automatiche: un betengine.db della v4 viene aggiornato
al primo avvio senza perdere dati.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "betengine.db")

# Colonne aggiunte dopo la creazione iniziale dello schema (v4 → v5).
# (nome, tipo SQL). La migrazione le aggiunge se mancano.
_MIGRATIONS = [
    ("prob_market", "REAL"),
    ("engine", "TEXT DEFAULT 'v4'"),
    ("prob_raw", "REAL"),
    ("shortlisted", "INTEGER DEFAULT 1"),
    ("sharp_book", "TEXT"),
    ("market_source", "TEXT"),
    ("odds_close", "REAL"),
    ("clv_pct", "REAL"),
    ("kelly_frac", "REAL"),
    ("mu_total", "REAL"),
    ("data_quality", "INTEGER"),
    ("degraded", "INTEGER DEFAULT 0"),
]


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
    """Crea la tabella se non esiste e migra i DB esistenti. Chiamata all'avvio."""
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
                prob_raw       REAL,
                prob_pure      REAL,
                prob_market    REAL,
                odds           REAL,
                odds_close     REAL,
                clv_pct        REAL,
                ev_pct         REAL,
                kelly_frac     REAL,
                mu_total       REAL,
                data_quality   INTEGER,
                degraded       INTEGER DEFAULT 0,
                stars          INTEGER,
                shortlisted    INTEGER DEFAULT 1,
                anchored       INTEGER DEFAULT 0,
                sharp_book     TEXT,
                market_source  TEXT,
                engine         TEXT DEFAULT 'v4',
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

        # --- Migrazione da schemi precedenti ---
        # DEVE avvenire PRIMA di creare indici sulle colonne nuove: su un DB
        # v4 la colonna shortlisted non esiste ancora e CREATE INDEX fallirebbe.
        existing = {row["name"] for row in
                    c.execute("PRAGMA table_info(predictions)").fetchall()}
        for col, coltype in _MIGRATIONS:
            if col not in existing:
                c.execute(f"ALTER TABLE predictions ADD COLUMN {col} {coltype}")

        c.execute("CREATE INDEX IF NOT EXISTS idx_short ON predictions(shortlisted)")

        # Righe della v4: prob_raw mancante. Le vecchie righe sono state
        # salvate quando la calibrazione era ancora inattiva (identità), quindi
        # prob == prob_raw ed è corretto ricopiarle. Le righe salvate CON
        # calibrazione attiva non esistono: la v4 non è mai arrivata a 150.
        c.execute("UPDATE predictions SET prob_raw = prob "
                  "WHERE prob_raw IS NULL AND prob IS NOT NULL")
        # Tutte le righe della v4 erano per definizione consigliate
        c.execute("UPDATE predictions SET shortlisted = 1 "
                  "WHERE shortlisted IS NULL")


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
                 prob, prob_raw, prob_pure, prob_market, odds, ev_pct,
                 kelly_frac, mu_total, data_quality, degraded,
                 stars, shortlisted, anchored, sharp_book,
                 market_source, engine)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                now, r.get("match_date"), r.get("fixture_id"),
                r.get("league_id"), r.get("league"),
                r.get("home"), r.get("away"), r.get("market"),
                r.get("selection"), r.get("selection_code"),
                r.get("prob"), r.get("prob_raw"), r.get("prob_pure"),
                r.get("prob_market"), r.get("odds"), r.get("ev_pct"),
                r.get("kelly_frac"), r.get("mu_total"),
                r.get("data_quality"),
                1 if r.get("degraded") else 0,
                r.get("stars"),
                1 if r.get("shortlisted") else 0,
                1 if r.get("anchored") else 0,
                r.get("sharp_book"), r.get("market_source"),
                r.get("engine", "v4"),
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


def predictions_awaiting_closing_odds(match_date: str) -> list:
    """
    Pronostici di una data specifica che hanno una quota registrata ma non
    ancora la quota di chiusura. Usato per il calcolo del CLV.
    """
    with _conn() as c:
        cur = c.execute("""
            SELECT * FROM predictions
            WHERE match_date = ? AND odds IS NOT NULL AND odds_close IS NULL
            ORDER BY fixture_id
        """, (match_date,))
        return [dict(row) for row in cur.fetchall()]


def set_closing_odds(pred_id: int, odds_close: float):
    """
    Registra la quota di chiusura e calcola il CLV.

    CLV% = (quota_presa / quota_chiusura - 1) × 100
    Positivo = hai preso una quota migliore di quella finale, cioè hai
    anticipato il mercato. È il segnale di edge più rapido da accumulare.
    """
    with _conn() as c:
        row = c.execute("SELECT odds FROM predictions WHERE id = ?",
                        (pred_id,)).fetchone()
        if not row or not row["odds"] or not odds_close or odds_close <= 1.0:
            return
        clv = (float(row["odds"]) / float(odds_close) - 1.0) * 100.0
        c.execute("UPDATE predictions SET odds_close = ?, clv_pct = ? "
                  "WHERE id = ?", (float(odds_close), round(clv, 2), pred_id))


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
                SUM(CASE WHEN status IN ('won','lost') THEN 1 ELSE 0 END) AS settled,
                SUM(CASE WHEN shortlisted = 1 THEN 1 ELSE 0 END) AS shortlisted,
                SUM(CASE WHEN clv_pct IS NOT NULL THEN 1 ELSE 0 END) AS with_clv,
                SUM(CASE WHEN degraded = 1 THEN 1 ELSE 0 END) AS degraded
            FROM predictions
        """)
        row = cur.fetchone()
        return {"total": row["total"] or 0,
                "pending": row["pending"] or 0,
                "settled": row["settled"] or 0,
                "shortlisted": row["shortlisted"] or 0,
                "with_clv": row["with_clv"] or 0,
                "degraded": row["degraded"] or 0}


def delete_all() -> int:
    """Svuota il database. Ritorna il numero di righe eliminate."""
    with _conn() as c:
        cur = c.execute("DELETE FROM predictions")
        return cur.rowcount


init_db()
