"""
🔧 SETUP ADMIN - Crea l'account amministratore
==============================================
Esegui questo script UNA VOLTA per creare il tuo account admin.

Uso:
    python setup_admin.py
"""

from database import create_user, init_database

def main():
    print("=" * 50)
    print("🔧 BetEngine - Setup Account Admin")
    print("=" * 50)
    print()
    
    # Inizializza database
    init_database()
    
    # Chiedi credenziali
    print("Inserisci le credenziali per l'account admin:")
    print()
    
    username = input("👤 Username: ").strip()
    if not username:
        print("❌ Username non può essere vuoto!")
        return
    
    password = input("🔒 Password: ").strip()
    if not password:
        print("❌ Password non può essere vuota!")
        return
    
    email = input("📧 Email (opzionale, premi Invio per saltare): ").strip()
    
    # Crea admin
    print()
    print("Creazione account admin in corso...")
    
    success = create_user(
        username=username,
        password=password,
        email=email if email else None,
        is_admin=True,
        subscription_days=36500  # 100 anni (praticamente infinito)
    )
    
    if success:
        print()
        print("=" * 50)
        print("✅ ACCOUNT ADMIN CREATO CON SUCCESSO!")
        print("=" * 50)
        print()
        print(f"   Username: {username}")
        print(f"   Email: {email if email else 'N/A'}")
        print(f"   Ruolo: Admin")
        print()
        print("Ora puoi avviare BetEngine con:")
        print("   streamlit run app.py")
        print()
    else:
        print()
        print("❌ ERRORE: Username già esistente!")
        print("   Prova con un username diverso.")


if __name__ == "__main__":
    main()
