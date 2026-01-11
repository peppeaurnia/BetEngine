"""
🗄️ DATABASE - Gestione utenti BetEngine
========================================
Database PostgreSQL (Supabase) per gestire utenti e abbonamenti.
"""

import hashlib
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import psycopg2
from psycopg2.extras import RealDictCursor

# Configurazione Supabase
SUPABASE_PASSWORD = os.environ.get("SUPABASE_PASSWORD", "ZS&Np7$f2hfa,VS")
SUPABASE_HOST = "db.xhryuvkqafobefefzqjr.supabase.co"
SUPABASE_DB = "postgres"
SUPABASE_USER = "postgres"
SUPABASE_PORT = "5432"


def get_connection():
    """Crea connessione al database PostgreSQL."""
    conn = psycopg2.connect(
        host=SUPABASE_HOST,
        database=SUPABASE_DB,
        user=SUPABASE_USER,
        password=SUPABASE_PASSWORD,
        port=SUPABASE_PORT
    )
    return conn


def hash_password(password: str) -> str:
    """Hash della password con SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def init_database():
    """Inizializza il database con la tabella utenti."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            email VARCHAR(255),
            is_admin INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            subscription_end TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    """)
    
    conn.commit()
    
    # Crea admin di default se non esiste nessun utente
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Crea utente admin di default
        admin_hash = hash_password("admin123")
        cursor.execute("""
            INSERT INTO users (username, password_hash, email, is_admin, is_active)
            VALUES (%s, %s, %s, 1, 1)
        """, ("Boppo", admin_hash, "admin@betengine.com"))
        conn.commit()
    
    cursor.close()
    conn.close()


def create_user(username: str, password: str, email: str = None, 
                is_admin: bool = False, subscription_days: int = 30) -> bool:
    """
    Crea un nuovo utente.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        subscription_end = datetime.now() + timedelta(days=subscription_days)
        
        cursor.execute("""
            INSERT INTO users (username, password_hash, email, is_admin, subscription_end)
            VALUES (%s, %s, %s, %s, %s)
        """, (username, hash_password(password), email, int(is_admin), subscription_end))
        
        conn.commit()
        return True
    except psycopg2.IntegrityError:
        conn.rollback()
        return False  # Username già esiste
    finally:
        cursor.close()
        conn.close()


def verify_user(username: str, password: str) -> Optional[Dict]:
    """
    Verifica credenziali utente.
    """
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT * FROM users 
        WHERE username = %s AND password_hash = %s AND is_active = 1
    """, (username, hash_password(password)))
    
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if row:
        return dict(row)
    return None


def check_subscription(username: str) -> bool:
    """Controlla se l'abbonamento è attivo."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT subscription_end, is_admin FROM users 
        WHERE username = %s AND is_active = 1
    """, (username,))
    
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not row:
        return False
    
    # Admin ha sempre accesso
    if row['is_admin']:
        return True
    
    # Controlla scadenza
    if row['subscription_end']:
        return datetime.now() < row['subscription_end']
    
    return False


def update_last_login(username: str):
    """Aggiorna timestamp ultimo login."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE users SET last_login = %s WHERE username = %s
    """, (datetime.now(), username))
    
    conn.commit()
    cursor.close()
    conn.close()


def get_all_users() -> List[Dict]:
    """Restituisce tutti gli utenti (per pannello admin)."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return [dict(row) for row in rows]


def extend_subscription(username: str, days: int) -> bool:
    """Estende l'abbonamento di X giorni."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Prendi la data attuale di scadenza
    cursor.execute("SELECT subscription_end FROM users WHERE username = %s", (username,))
    row = cursor.fetchone()
    
    if not row:
        cursor.close()
        conn.close()
        return False
    
    # Se scaduto, parti da oggi
    if row['subscription_end']:
        current_end = row['subscription_end']
        if current_end < datetime.now():
            current_end = datetime.now()
    else:
        current_end = datetime.now()
    
    new_end = current_end + timedelta(days=days)
    
    cursor.execute("""
        UPDATE users SET subscription_end = %s WHERE username = %s
    """, (new_end, username))
    
    conn.commit()
    cursor.close()
    conn.close()
    return True


def deactivate_user(username: str) -> bool:
    """Disattiva un utente."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("UPDATE users SET is_active = 0 WHERE username = %s", (username,))
    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()
    
    return affected > 0


def activate_user(username: str) -> bool:
    """Riattiva un utente."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("UPDATE users SET is_active = 1 WHERE username = %s", (username,))
    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()
    
    return affected > 0


def delete_user(username: str) -> bool:
    """Elimina un utente (permanente)."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM users WHERE username = %s AND is_admin = 0", (username,))
    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()
    
    return affected > 0


def change_password(username: str, new_password: str) -> bool:
    """Cambia la password di un utente."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE users SET password_hash = %s WHERE username = %s
    """, (hash_password(new_password), username))
    
    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()
    
    return affected > 0


# Inizializza il database all'import
init_database()
