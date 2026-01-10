"""
🔐 AUTH - Sistema di autenticazione BetEngine
=============================================
Gestisce login, logout e sessioni utente.
"""

import streamlit as st
from database import (
    verify_user, 
    check_subscription, 
    update_last_login,
    get_all_users,
    create_user,
    extend_subscription,
    deactivate_user,
    activate_user,
    delete_user,
    change_password
)
from datetime import datetime


def init_session():
    """Inizializza le variabili di sessione."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False


def login(username: str, password: str) -> tuple[bool, str]:
    """
    Effettua il login.
    
    Returns:
        (success: bool, message: str)
    """
    user = verify_user(username, password)
    
    if not user:
        return False, "❌ Username o password errati"
    
    # Controlla abbonamento
    if not user['is_admin'] and not check_subscription(username):
        return False, "⚠️ Il tuo abbonamento è scaduto. Contatta l'amministratore."
    
    # Login riuscito
    st.session_state.authenticated = True
    st.session_state.username = username
    st.session_state.is_admin = bool(user['is_admin'])
    
    update_last_login(username)
    
    return True, "✅ Login effettuato!"


def logout():
    """Effettua il logout."""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.is_admin = False


def is_authenticated() -> bool:
    """Controlla se l'utente è autenticato."""
    return st.session_state.get('authenticated', False)


def is_admin() -> bool:
    """Controlla se l'utente è admin."""
    return st.session_state.get('is_admin', False)


def get_current_user() -> str:
    """Restituisce username corrente."""
    return st.session_state.get('username', None)


def show_login_page():
    """Mostra la pagina di login."""
    
    # CSS per la pagina login (tema scuro)
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Audiowide&display=swap');
        
        /* Sfondo scuro */
        .stApp {
            background: linear-gradient(180deg, #1a3a52 0%, #0d2137 100%);
        }
        
        .login-subtitle {
            text-align: center;
            color: #b0c4d8;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }
        
        .login-box {
            max-width: 400px;
            margin: 0 auto;
            padding: 30px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        /* Input visibili */
        .stTextInput > div > div > input {
            background-color: #ffffff !important;
            color: #1a1a2e !important;
        }
        
        /* Label bianche per il form login */
        .stTextInput label,
        .stTextInput label p,
        .stTextInput label span,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span {
            color: #ffffff !important;
        }
        
        /* Testo nero nelle box bianche */
        .stAlert, .stAlert p, .stAlert span, .stAlert div {
            color: #1a1a2e !important;
        }
        
        .stForm label {
            color: #ffffff !important;
        }
        
        /* Nascondi elementi Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stHeader"] {display: none;}
        [data-testid="stToolbar"] {display: none;}
        
        /* Nascondi pulsante fullscreen immagini */
        button[title="View fullscreen"] {display: none;}
        [data-testid="StyledFullScreenButton"] {display: none;}
    </style>
    """, unsafe_allow_html=True)
    
    # Logo centrato
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        try:
            import base64
            with open("logo.png", "rb") as f:
                logo_data = base64.b64encode(f.read()).decode()
            st.markdown(f"""
            <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
                <img src="data:image/png;base64,{logo_data}" style="width: 220px;">
            </div>
            """, unsafe_allow_html=True)
        except:
            st.image("logo.png", width=220)
    
    st.markdown('<p class="login-subtitle">Trasforma i dati in probabilità vincenti</p>', unsafe_allow_html=True)
    
    # Form login centrato
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            username = st.text_input("👤 Username")
            password = st.text_input("🔒 Password", type="password")
            
            submit = st.form_submit_button("🚀 ACCEDI", use_container_width=True)
            
            if submit:
                if username and password:
                    success, message = login(username, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Inserisci username e password")
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #888; font-size: 0.85rem;">
            <p>Non hai un account?</p>
            <p>Contatta l'amministratore per abbonarti</p>
        </div>
        """, unsafe_allow_html=True)


def show_admin_panel():
    """Mostra il pannello di amministrazione."""
    
    # CSS specifico per il pannello admin
    st.markdown("""
    <style>
        /* Label bianche SOLO nel pannello admin */
        .stTabs [data-baseweb="tab-panel"] label,
        .stTabs [data-baseweb="tab-panel"] label p,
        .stTabs [data-baseweb="tab-panel"] label span,
        .stTabs [data-baseweb="tab-panel"] .stTextInput > label,
        .stTabs [data-baseweb="tab-panel"] .stNumberInput > label,
        .stTabs [data-baseweb="tab-panel"] .stCheckbox > label,
        .stTabs [data-baseweb="tab-panel"] .stSelectbox > label,
        .stTabs [data-baseweb="tab-panel"] [data-testid="stWidgetLabel"] p {
            color: #ffffff !important;
        }
        
        /* Bottone "Crea Utente" - testo nero */
        .stTabs [data-baseweb="tab-panel"] .stButton button p,
        .stTabs [data-baseweb="tab-panel"] .stButton button span {
            color: #000000 !important;
        }
        
        /* Messaggio info "Nessun utente da gestire" - testo bianco */
        .stTabs [data-baseweb="tab-panel"] .stAlert p,
        .stTabs [data-baseweb="tab-panel"] .stAlert span,
        .stTabs [data-baseweb="tab-panel"] [data-testid="stAlert"] p {
            color: #ffffff !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## 👑 Pannello Admin")
    
    tab1, tab2, tab3 = st.tabs(["📋 Utenti", "➕ Nuovo Utente", "⚙️ Gestione"])
    
    with tab1:
        st.markdown("### Lista Utenti")
        users = get_all_users()
        
        if users:
            for user in users:
                # Calcola stato abbonamento
                if user['is_admin']:
                    status = "👑 Admin"
                    status_color = "#9b59b6"
                elif not user['is_active']:
                    status = "🚫 Disattivato"
                    status_color = "#e74c3c"
                elif user['subscription_end']:
                    try:
                        end_date = datetime.strptime(user['subscription_end'], '%Y-%m-%d %H:%M:%S.%f')
                        if datetime.now() < end_date:
                            days_left = (end_date - datetime.now()).days
                            status = f"✅ Attivo ({days_left} giorni)"
                            status_color = "#27ae60"
                        else:
                            status = "⚠️ Scaduto"
                            status_color = "#f39c12"
                    except:
                        status = "❓ Sconosciuto"
                        status_color = "#95a5a6"
                else:
                    status = "❓ Sconosciuto"
                    status_color = "#95a5a6"
                
                with st.expander(f"**{user['username']}** - {status}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"📧 Email: {user['email'] or 'N/A'}")
                        st.write(f"📅 Creato: {user['created_at'][:10] if user['created_at'] else 'N/A'}")
                    with col2:
                        st.write(f"🕐 Ultimo login: {user['last_login'][:16] if user['last_login'] else 'Mai'}")
                        st.write(f"📆 Scadenza: {user['subscription_end'][:10] if user['subscription_end'] else 'N/A'}")
        else:
            st.info("Nessun utente registrato")
    
    with tab2:
        st.markdown("### Crea Nuovo Utente")
        
        with st.form("new_user_form"):
            new_username = st.text_input("Username *")
            new_password = st.text_input("Password *", type="password")
            new_email = st.text_input("Email")
            sub_days = st.number_input("Giorni abbonamento", min_value=1, value=30)
            make_admin = st.checkbox("È admin?")
            
            if st.form_submit_button("✅ Crea Utente", use_container_width=True):
                if new_username and new_password:
                    if create_user(new_username, new_password, new_email, make_admin, sub_days):
                        st.success(f"✅ Utente '{new_username}' creato con successo!")
                        st.rerun()
                    else:
                        st.error("❌ Username già esistente")
                else:
                    st.warning("Username e password sono obbligatori")
    
    with tab3:
        st.markdown("### Gestione Utenti")
        
        users = get_all_users()
        non_admin_users = [u['username'] for u in users if not u['is_admin']]
        
        if non_admin_users:
            selected_user = st.selectbox("Seleziona utente", non_admin_users)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                days_to_add = st.number_input("Giorni da aggiungere", min_value=1, value=30)
                if st.button("➕ Estendi Abbonamento"):
                    if extend_subscription(selected_user, days_to_add):
                        st.success(f"✅ Aggiunti {days_to_add} giorni a {selected_user}")
                        st.rerun()
            
            with col2:
                if st.button("🚫 Disattiva Utente"):
                    if deactivate_user(selected_user):
                        st.success(f"✅ {selected_user} disattivato")
                        st.rerun()
                
                if st.button("✅ Riattiva Utente"):
                    if activate_user(selected_user):
                        st.success(f"✅ {selected_user} riattivato")
                        st.rerun()
            
            with col3:
                new_pwd = st.text_input("Nuova password", type="password", key="new_pwd")
                if st.button("🔑 Cambia Password"):
                    if new_pwd:
                        if change_password(selected_user, new_pwd):
                            st.success(f"✅ Password cambiata per {selected_user}")
                    else:
                        st.warning("Inserisci la nuova password")
                
                if st.button("🗑️ Elimina Utente", type="secondary"):
                    if delete_user(selected_user):
                        st.success(f"✅ {selected_user} eliminato")
                        st.rerun()
        else:
            st.info("Nessun utente da gestire")


def show_user_info_sidebar():
    """Mostra info utente nella sidebar."""
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<span style='color:white !important; font-weight:bold;'>👤 {get_current_user()}</span>", unsafe_allow_html=True)
    
    if is_admin():
        st.sidebar.markdown("<span style='color:white !important;'>👑 Admin</span>", unsafe_allow_html=True)
    
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        logout()
        st.rerun()
