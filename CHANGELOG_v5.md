# BetEngine v5 — note di rilascio

## Come installare

Sostituisci i file mantenendo la struttura:

```
progetto/
├── app.py                    ← sostituire
├── probability_engine.py     ← sostituire
├── storage.py                ← sostituire
├── calibration.py            ← sostituire
├── odds_api.py               ← sostituire
├── dc_model.py               ← sostituire
├── results_updater.py        ← sostituire
├── estimate_probabilities.py ← sostituire
├── requirements.txt          ← sostituire
├── config.py                 ← INVARIATO
├── data_fetcher.py           ← INVARIATO
├── fetch_referee_stats.py    ← INVARIATO
└── pages/
    └── 1_Performance.py      ← sostituire
```

Il database `betengine.db` esistente viene migrato automaticamente al primo
avvio: nessuna perdita di dati, nessuna azione manuale. **Fai comunque una
copia di `betengine.db` prima**, per abitudine.

`calibration.json`, se esiste, va **cancellato**: è stato stimato con il
vecchio metodo su `prob`. Si rigenera dalla pagina Performance.

Nessuna dipendenza nuova.

---

## 1. Selezione per valore atteso invece che per probabilità

La v4 consigliava un mercato se la sua probabilità superava una soglia (55%
per 1X2, 65% per O/U e BTTS, 78% per i cartellini) e mostrava l'EV come
decorazione. È il criterio sbagliato: un 78% a quota 1.20 ha EV −6%, un 30% a
quota 4.50 ha EV +35%. Peggio, le soglie alte spingevano sistematicamente sui
grandi favoriti e sugli Over cartellini, dove il margine del bookmaker è più
alto.

Ora il gate primario è l'EV (`MIN_EV_PCT = 4.0`), con la probabilità come
filtro di sanità (`MIN_PROB_SANITY = 22.0`) e un tetto sulla quota
(`MAX_ODDS = 8.0`) per non inseguire longshot.

Esempio reale dal test di integrazione, stessa partita:

| criterio | selezione | probabilità | EV |
|---|---|---|---|
| v4 (probabilità) | 1 | 57.9% | **+0.3%** |
| v5 (valore) | O2.5 | 53.7% | **+11.2%** |

Le stelle ora misurano il valore, non la probabilità: cinque pallini su un 85%
a quota 1.05 erano una bugia visiva.

**Staking**: aggiunto Kelly frazionario (1/4, tetto 2% del bankroll). Kelly
pieno è ottimale solo con probabilità esatte; con probabilità stimate porta a
rovina.

Se non ci sono quote, l'app ripiega sulle vecchie soglie **e lo dichiara
esplicitamente**: un pronostico senza prezzo non è una scommessa, è
un'opinione.

---

## 2. Quote sharp separate dalle quote migliori

`fetch_match_odds` restituiva la quota **massima per esito su tutti i
bookmaker**, e quel dizionario veniva usato per due cose incompatibili:
calcolare l'EV (corretto) e stimare le probabilità implicite del mercato
(sbagliato).

Prendere il massimo esito per esito costruisce un book sintetico che nessuno
ha mai quotato: il margine risulta già eroso e la *forma* della distribuzione
è distorta. Quell'artefatto finiva sia nel market anchoring sia nel benchmark
Brier che dovrebbe dire se il modello ha un edge — rendendolo artificialmente
facile da battere.

`odds_api.py` ora restituisce:
- `best` — massimo per esito → EV, staking, display
- `sharp` — quote di **un solo** book (Pinnacle → exchange → book col margine
  più basso) → probabilità di mercato

L'app mostra quale book ha fornito le probabilità, e segnala in arancione
quando ha dovuto ripiegare su un book non sharp.

**Bonus**: una sola chiamata API per partita invece di tre (i tre mercati si
chiedono insieme).

---

## 3. Rimozione del margine col metodo di Shin

La normalizzazione proporzionale (`p = (1/o) / Σ(1/o)`) assume che il
bookmaker spalmi il margine uniformemente, mentre lo concentra sugli outsider.
Risultato: favoriti sottostimati, outsider sovrastimati, in modo sistematico.

Sostituita da Shin (1993) in `probability_engine._shin_probs`, usata sia
nell'app sia nel backtest. Verifica su terna 1.50 / 4.20 / 7.00
(booksum 1.0476):

| esito | proporzionale | Shin | delta |
|---|---|---|---|
| 1 | 63.64% | 64.71% | **+1.07** |
| X | 22.73% | 22.34% | −0.38 |
| 2 | 13.64% | 12.95% | −0.69 |

Con margine nullo il metodo degenera nella normalizzazione proporzionale:
nel caso peggiore non si perde nulla.

---

## 4. Loop di feedback nella calibrazione — chiuso

L'app salvava in `prob` la probabilità **già calibrata**, e
`calibration.fit_and_save()` fittava la curva su quella colonna. Finché la
calibrazione era inattiva non si notava (`apply()` era l'identità), ma al
primo refit dopo l'attivazione la curva veniva stimata su valori già corretti
e riapplicata alle probabilità grezze: la correzione si componeva su sé stessa
a ogni ricalcolo.

Ora si salvano tre colonne distinte:
- `prob_raw` — ancorata ma **non** calibrata → input del fit
- `prob` — quello che l'app ha mostrato → display e audit
- `prob_pure` — modello puro, senza anchoring → EV

Verificato nel test: dopo due `fit_and_save()` consecutivi, `apply(70)`
restituisce lo stesso identico valore.

**Due curve, non una.** `prob_raw` e `prob_pure` hanno distribuzioni diverse e
quindi errori di calibrazione diversi: applicare la curva dell'una all'altra è
un errore sistematico invisibile. Ora si stimano separatamente
(`apply(..., kind="raw"|"pure")`).

---

## 5. Si salvano tutti i mercati, non solo i consigliati

Prima finivano nel database solo i 2-3 consigliati, tutti sopra soglia: la
curva di calibrazione era stimabile solo nella coda alta, il Brier era
misurato su un campione censurato, e per arrivare a 150 esiti servivano mesi.

Ora ogni mercato valutato viene salvato, con `shortlisted` a marcare quelli
proposti. Nel test: **23 righe per partita invece di 3** (7.7×). La soglia dei
150 esiti arriva in settimane invece che in mesi.

La pagina Performance ha un selettore *Solo consigliati / Tutti i mercati*,
perché sono due domande diverse: win rate e yield hanno senso solo sui
consigliati (le scommesse che avresti davvero fatto), calibrazione e Brier
sono più informativi su tutto il campione.

La curva di calibrazione ora grafica `prob_raw`: graficare `prob` la
metterebbe sulla diagonale per costruzione, nascondendo l'errore da misurare.

---

## 6. CLV (Closing Line Value)

Nuove colonne `odds_close` e `clv_pct`, nuova funzione
`results_updater.update_closing_odds()`, pulsante dedicato in Performance.

Lo yield ha bisogno di centinaia di esiti per staccarsi dal rumore; il CLV dà
un segnale leggibile con 40-50 scommesse, perché misura il confronto diretto
con il mercato invece di passare attraverso l'esito casuale della singola
partita. Uno yield positivo con CLV negativo è quasi sempre fortuna.

**Va lanciato vicino al calcio d'inizio**: è nell'ultima ora che il mercato
incorpora formazioni e ultime notizie.

---

## 7. Cartellini: binomiale negativa

I cartellini sono sovradispersi (varianza ≈ 1.3-1.4× la media). Poisson impone
varianza = media e comprime le code. Effetto pratico del bug: Under troppo
alti sulle linee basse, proposti come "sicuri al 78%" quando la frequenza
reale è più bassa — esattamente il tipo di errore che produce value bet
fantasma su un mercato con margine alto.

Con λ = 5.0:

| linea | Poisson | NegBin | delta |
|---|---|---|---|
| O3.5 | 73.5% | 69.3% | **−4.2** |
| O4.5 | 56.0% | 53.4% | −2.6 |
| O5.5 | 38.4% | 38.3% | −0.1 |

`CARDS_VMR = 1.35` è un valore centrale prudente, da ristimare dai tuoi dati
quando avrai abbastanza esiti con `total_cards` registrati.

---

## 8. Dixon-Coles: ξ stimato, non indovinato

`XI = 0.0019` (emivita ~1 anno) era fissato a mano ed era il parametro più
influente del modello. Ora viene scelto per **validazione temporale**: fit
sulle partite più vecchie, log-verosimiglianza misurata sulle più recenti mai
viste. È il criterio corretto perché è esattamente il compito che il modello
svolge in produzione.

```
python dc_model.py fit-all              # con tuning (default)
python dc_model.py fit-all --no-tune    # salta il tuning
```

Costa ~30-60 secondi per lega, **zero chiamate API**.

**Parità di shrinkage.** I μ del DC ora ricevono lo stesso shrinkage
James-Stein della v4. Prima non era così: il DC produceva probabilità più
estreme, superava le soglie più spesso e generava più pronostici, quindi il
confronto A/B misurava insieme accuratezza e aggressività. Ora la differenza
residua è attribuibile al modello.

La tabella A/B in Performance ha due colonne nuove: **Brier mercato** (sulle
stesse righe) e **CLV**. Due motori possono essere entrambi peggiori del
mercato: confrontarli solo tra loro lo nasconderebbe.

---

## 9. Doppio conteggio forma/momentum

`SOFT_ADJ_WEIGHT` da 0.25 a 0.12. `form_factor` e `momentum` sono entrambi
costruiti dai punti delle ultime partite: non sono due segnali indipendenti da
mediare, sono lo stesso segnale contato due volte — e quelle partite sono già
dentro i totali stagionali da cui si ricavano le strength.

Conta solo per il fallback v4: il Dixon-Coles risolve il problema alla radice
col decadimento temporale.

---

## Cosa NON è stato fatto, e perché

**xG.** API-Football non espone gli xG su `/teams/statistics`, solo su
`/fixtures/statistics` — una chiamata per partita, ~380 per lega-stagione,
fuori portata anche col piano Pro. Scrivere codice speculativo su una risposta
API non verificabile sarebbe peggio che non scriverlo. Resta il miglioramento
con più potenziale, ma richiede prima una decisione sulla fonte dati (Understat
o FBref hanno xG gratuiti, con il costo di un secondo mapping dei nomi squadra).

**Aggiustamento per forza degli avversari nel motore v4.** Le strength restano
medie stagionali grezze. È il problema che il Dixon-Coles risolve: la strada
giusta è ridurre l'uso del fallback v4, non rattoppare il fallback.

**Chiavi API.** Lasciate come richiesto. Restano in chiaro in `config.py`.

---

## Test eseguiti

- Metodo di Shin: somma a 1 esatta, direzione della correzione verificata
- Binomiale negativa: somma a 1 su 0-60, code confrontate con Poisson
- Fit Dixon-Coles su campionato sintetico: γ stimato 0.2641 contro vero 0.26
- `tune_xi`: massimo interno pulito sulla griglia, nessun plateau
- Migrazione DB v4→v5: 30 righe preservate, `prob_raw` copiata da `prob`,
  idempotente su reinizializzazione
- Ciclo completo: 40 partite → 920 righe salvate → duplicati ignorati (0
  reinserimenti) → 920 esiti chiusi → calibrazione attiva con curve per
  mercato e curva pura
- Monotonia della curva isotonica verificata
- **Stabilità dopo refit**: `apply(70)` identico prima e dopo un secondo fit
  (loop di feedback chiuso)
- `determine_outcome` su 14 casi incluse le nuove linee cartellini: 0 errori
- CLV: quota 2.10 → chiusura 1.95 → +7.69% (atteso +7.69%)
- `1_Performance.py` eseguita headless su DB realistico (690 righe, due
  engine, 210 con CLV) in entrambi gli scope

---

## Primo giro consigliato

1. Copia di sicurezza di `betengine.db`, poi sostituisci i file
2. Cancella `calibration.json`
3. Avvia l'app: la migrazione parte da sola
4. `python dc_model.py fit-all` (il tuning di ξ ti dirà quanto era sbagliato
   0.0019 sulle tue leghe)
5. Analizza qualche partita e salva: dovresti vedere ~20 mercati salvati per
   partita e 0-3 consigliati
6. Vicino al calcio d'inizio, registra le quote di chiusura
7. Dopo le partite, aggiorna i risultati

Il criterio per iniziare a rischiare soldi resta quello: **Brier del modello
sotto il Brier del mercato su 300+ partite, e CLV medio positivo**. Fino ad
allora è raccolta dati.
