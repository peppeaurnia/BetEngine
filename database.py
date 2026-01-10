"""
🗄️ DATABASE - Gestione utenti BetEngine
========================================
Database SQLite per gestire utenti e abbonamenti.
"""

import sqlite3
import hashlib
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List

# Path del database
DB_PATH = "betengine_users.db"


def get_connection():
    """Crea connessione al database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Inizializza il database con la tabella utenti."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            is_admin INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            subscription_end DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    """Hash della password con SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username: str, password: str, email: str = None, 
                is_admin: bool = False, subscription_days: int = 30) -> bool:
    """
    Crea un nuovo utente.
    
    Args:
        username: Nome utente
        password: Password in chiaro (verrà hashata)
        email: Email opzionale
        is_admin: Se è admin
        subscription_days: Giorni di abbonamento (default 30)
    
    Returns:
        True se creato, False se username già esiste
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        subscription_end = datetime.now() + timedelta(days=subscription_days)
        
        cursor.execute("""
            INSERT INTO users (username, password_hash, email, is_admin, subscription_end)
            VALUES (?, ?, ?, ?, ?)
        """, (username, hash_password(password), email, int(is_admin), subscription_end))
        
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username già esiste
    finally:
        conn.close()


def verify_user(username: str, password: str) -> Optional[Dict]:
    """
    Verifica credenziali utente.
    
    Returns:
        Dict con dati utente se valido, None se credenziali errate
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM users 
        WHERE username = ? AND password_hash = ? AND is_active = 1
    """, (username, hash_password(password)))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def check_subscription(username: str) -> bool:
    """Controlla se l'abbonamento è attivo."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT subscription_end, is_admin FROM users 
        WHERE username = ? AND is_active = 1
    """, (username,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return False
    
    # Admin ha sempre accesso
    if row['is_admin']:
        return True
    
    # Controlla scadenza
    if row['subscription_end']:
        end_date = datetime.strptime(row['subscription_end'], '%Y-%m-%d %H:%M:%S.%f')
        return datetime.now() < end_date
    
    return False


def update_last_login(username: str):
    """Aggiorna timestamp ultimo login."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE users SET last_login = ? WHERE username = ?
    """, (datetime.now(), username))
    
    conn.commit()
    conn.close()


def get_all_users() -> List[Dict]:
    """Restituisce tutti gli utenti (per pannello admin)."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def extend_subscription(username: str, days: int) -> bool:
    """Estende l'abbonamento di X giorni."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Prendi la data attuale di scadenza
    cursor.execute("SELECT subscription_end FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return False
    
    # Se scaduto, parti da oggi
    if row['subscription_end']:
        try:
            current_end = datetime.strptime(row['subscription_end'], '%Y-%m-%d %H:%M:%S.%f')
            if current_end < datetime.now():
                current_end = datetime.now()
        except:
            current_end = datetime.now()
    else:
        current_end = datetime.now()
    
    new_end = current_end + timedelta(days=days)
    
    cursor.execute("""
        UPDATE users SET subscription_end = ? WHERE username = ?
    """, (new_end, username))
    
    conn.commit()
    conn.close()
    return True


def deactivate_user(username: str) -> bool:
    """Disattiva un utente."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("UPDATE users SET is_active = 0 WHERE username = ?", (username,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    
    return affected > 0


def activate_user(username: str) -> bool:
    """Riattiva un utente."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("UPDATE users SET is_active = 1 WHERE username = ?", (username,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    
    return affected > 0


def delete_user(username: str) -> bool:
    """Elimina un utente (permanente)."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM users WHERE username = ? AND is_admin = 0", (username,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    
    return affected > 0


def change_password(username: str, new_password: str) -> bool:
    """Cambia la password di un utente."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE users SET password_hash = ? WHERE username = ?
    """, (hash_password(new_password), username))
    
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    
    return affected > 0


# Inizializza il database all'import
init_database()
