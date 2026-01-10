# ⚽ Match Probability Predictor

> Calcola le probabilità di un match di calcio usando la distribuzione Poisson Bivariata con correzione Dixon-Coles.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Indice

- [Caratteristiche](#-caratteristiche)
- [Installazione](#-installazione)
- [Uso](#-uso)
- [Come Funziona](#-come-funziona)
- [Struttura Files](#-struttura-files)
- [API Reference](#-api-reference)
- [Limitazioni](#-limitazioni)

---

## ✨ Caratteristiche

- **🎲 Probabilità 1X2**: Vittoria casa, pareggio, vittoria trasferta
- **📊 Over/Under**: Linee 1.5, 2.5, 3.5, 4.5
- **⚽ BTTS (Gol/NoGol)**: Both Teams To Score
- **🎯 Punteggi Esatti**: Top 10 risultati più probabili
- **🔢 Matrice Completa**: Visualizzazione heatmap di tutti i punteggi
- **📈 Indicatore Qualità**: Affidabilità della previsione basata sui dati

### Leghe Supportate

| Lega | ID | λ₃ Calibrato |
|------|-----|--------------|
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League | 39 | 0.08 |
| 🇮🇹 Serie A | 135 | 0.05 |
| 🇪🇸 LaLiga | 140 | 0.05 |
| 🇩🇪 Bundesliga | 78 | 0.10 |
| 🇫🇷 Ligue 1 | 61 | 0.08 |
| 🇵🇹 Primeira Liga | 94 | 0.08 |
| 🇳🇱 Eredivisie | 88 | 0.08 |
| 🏆 Champions League | 2 | 0.08 |
| 🏆 Europa League | 3 | 0.08 |

---

## 🚀 Installazione

### Prerequisiti

- Python 3.9 o superiore
- API Key di [API-Football](https://www.api-football.com/) (piano gratuito: 100 chiamate/giorno)

### Passaggi

1. **Clona o scarica i files**
```bash
mkdir match_predictor
cd match_predictor
# Copia tutti i files .py qui
```

2. **Crea ambiente virtuale (consigliato)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure
venv\Scripts\activate     # Windows
```

3. **Installa dipendenze**
```bash
pip install -r requirements.txt
```

4. **Avvia l'applicazione**
```bash
streamlit run app.py
```

5. **Apri il browser** su `http://localhost:8501`

---

## 💻 Uso

### Passo 1: Inserisci API Key
- Nella sidebar, inserisci la tua chiave API di API-Football
- L'app verificherà automaticamente la connessione

### Passo 2: Seleziona la Lega
- Scegli il campionato dal menu dropdown

### Passo 3: Seleziona le Squadre
- **Squadra Casa**: La squadra che gioca in casa
- **Squadra Trasferta**: La squadra ospite

### Passo 4: Calcola
- Premi "🔮 CALCOLA PROBABILITÀ"
- Attendi il recupero dei dati (pochi secondi)
- Visualizza tutti i risultati!

### Refresh Dati
- Usa il pulsante "🔄 Refresh Dati" nella sidebar per pulire la cache
- Utile se le statistiche sono cambiate (nuove partite giocate)

---

## 🧮 Come Funziona

### Il Modello Matematico

L'applicazione usa la **Distribuzione Poisson Bivariata** con la correzione **Dixon-Coles** per modellare la correlazione tra i gol delle due squadre.

#### 1. Calcolo Expected Goals (μ)

Per ogni squadra calcoliamo i gol attesi:

```
μ_home = media_lega_gf_home × attack_strength_home × defense_strength_away × soft_adjustment
μ_away = media_lega_gf_away × attack_strength_away × defense_strength_home × soft_adjustment
```

Dove:
- **attack_strength** = gol_fatti_squadra / media_lega (range 0.55-1.60)
- **defense_strength** = gol_subiti_squadra / media_lega (range 0.55-1.60)
- **soft_adjustment** = combinazione di forma, classifica e momentum

#### 2. Correlazione Dixon-Coles (λ₃)

Il parametro λ₃ cattura la correlazione tra i gol:
- **Valori alti** (es. 0.10): Partite "aperte" - se una squadra segna, è più probabile che anche l'altra segni
- **Valori bassi** (es. 0.05): Partite tattiche - meno correlazione

#### 3. Matrice Poisson Bivariata

```
P(home=i, away=j) = Σ_k [P_poisson(i-k, λ₁) × P_poisson(j-k, λ₂) × P_poisson(k, λ₃)]
```

Dove:
- λ₁ = μ_home - λ₃
- λ₂ = μ_away - λ₃

#### 4. Calcolo Probabilità

Dalla matrice M[i,j] estraiamo:

| Esito | Formula |
|-------|---------|
| **P(1)** Vittoria Casa | Somma triangolo superiore (home > away) |
| **P(X)** Pareggio | Somma diagonale (home = away) |
| **P(2)** Vittoria Trasferta | Somma triangolo inferiore (away > home) |
| **P(Over 2.5)** | Somma celle dove i+j ≥ 3 |
| **P(BTTS Yes)** | Somma celle dove i ≥ 1 AND j ≥ 1 |

### Fonti Dati

I dati provengono da **API-Football** e includono:
- Statistiche stagione corrente (peso 70%)
- Statistiche stagione precedente (peso 30%)
- Forma ultime 5 partite
- Posizione in classifica
- Momentum (punti ultime 5 partite)

---

## 📁 Struttura Files

```
match_predictor/
├── app.py                    # Applicazione Streamlit (UI)
├── probability_engine.py     # Motore calcolo probabilità
├── data_fetcher.py           # Recupero dati da API
├── requirements.txt          # Dipendenze Python
└── README.md                 # Documentazione
```

### Descrizione Moduli

#### `probability_engine.py`
Contiene tutta la matematica:
- `poisson_pmf()`: Funzione massa Poisson
- `bivariate_poisson_matrix()`: Costruisce matrice probabilità
- `calculate_1x2()`: Calcola probabilità 1X2
- `calculate_over_under()`: Calcola O/U per ogni linea
- `calculate_btts()`: Calcola BTTS
- `calculate_match_probabilities()`: Funzione principale

#### `data_fetcher.py`
Gestisce le chiamate API:
- `fetch_teams_for_league()`: Lista squadre
- `fetch_team_statistics()`: Stats squadra
- `fetch_standings()`: Classifica
- `get_match_stats()`: Raccoglie tutti i dati per un match

#### `app.py`
Interfaccia utente Streamlit:
- Selezione lega/squadre
- Visualizzazione risultati
- Grafici Plotly

---

## 🔌 API Reference

### API-Football Endpoints Utilizzati

| Endpoint | Descrizione | Cache |
|----------|-------------|-------|
| `/teams` | Lista squadre per lega | 1 ora |
| `/teams/statistics` | Statistiche squadra | 30 min |
| `/standings` | Classifica | 1 ora |
| `/fixtures` | Ultime partite (momentum) | 30 min |
| `/status` | Test connessione | No cache |

### Rate Limits

- **Piano Free**: 100 chiamate/giorno
- **Piano Basic**: 7,500 chiamate/giorno

L'app usa caching aggressivo per minimizzare le chiamate.

---

## ⚠️ Limitazioni

1. **Non è consulenza finanziaria**: Le probabilità sono stime statistiche, non certezze
2. **Dati storici**: Il modello si basa su dati passati che potrebbero non riflettere cambiamenti recenti
3. **Fattori non considerati**:
   - Infortuni giocatori chiave
   - Condizioni meteo
   - Motivazioni stagionali
   - Scontri diretti
4. **Nuove squadre**: Squadre neopromosse potrebbero avere dati insufficienti
5. **API Limits**: Piano gratuito limitato a 100 chiamate/giorno

---

## 📊 Interpretazione Risultati

### Indicatore Qualità

| Score | Livello | Significato |
|-------|---------|-------------|
| 85-100% | 🟢 Alta | Dati completi e affidabili |
| 65-84% | 🟡 Media | Alcuni dati mancanti |
| <65% | 🔴 Bassa | Molti dati mancanti |

### Quote Implicite

Le "quote implicite" mostrate sono calcolate come:
```
Quota = 100 / Probabilità%
```

Rappresentano la quota "fair" senza margine del bookmaker.

---

## 🤝 Crediti

- **Modello Statistico**: Basato su ricerche di Dixon & Coles (1997)
- **Dati**: [API-Football](https://www.api-football.com/)
- **Framework**: [Streamlit](https://streamlit.io/)
- **Visualizzazioni**: [Plotly](https://plotly.com/)

---

## 📜 Licenza

MIT License - Sentiti libero di usare e modificare!

---

*Sviluppato con ❤️ per il betting intelligente*
