# Revisione finale della relazione — 23 ottobre 2026

## Esito metodologico

Non è stata rilevata un'incoerenza che richieda modifiche al simulatore o
nuovi esperimenti. Il principale chiarimento riguarda due disponibilità
temporali differenti: feature/candidati da D_<t e feedback maturi a scadenza
<= t. La seconda può includere l'ascolto al bordo destro esattamente t.
Non va quindi estesa all'intera memoria del learner la formula dello stato.
La convenzione dà priorità ai feedback maturi, poi alle decisioni,
poi all'ingresso degli eventi simultanei nello stato storico.

Le osservazioni su «preregistrata», giustificazione categorica della
macro-media, leggibilità dei grafici e paragrafo Global Popularity spezzato
erano correzioni effettive del testo o dell'impaginazione. Formule delle
metriche, assenza di apprendimento nel warm-up e soglie erano già coerenti
con il codice, ma meritavano spiegazioni ed esempi più espliciti.

## Evidenze dirette nel codice, senza eseguire policy

| Punto | Funzioni e comportamento verificati |
| --- | --- |
| Stato | src/data/state.py, TemporalState.advance_to() (121): consuma solo eventi con timestamp < t. |
| Ordine temporale | src/simulation/simulator.py, run_phase() (460): reset, coda iniziale vuota, filtro di fase/orizzonte; _process_pending(t) prima di state.advance_to(t), poi tutte le decisioni del gruppo, infine avanzamento a t+1 microsecondo. |
| Feedback al bordo | _process_pending() (433) usa disponibilità <= t; src/simulation/reward.py, reward_for_artist_key() (190) usa bisect_right su entrambi i limiti: intervallo (s,s+7d]. L'ascolto a t=s+7d può dunque entrare in A,b prima della scelta a t. |
| Test esistente | tests/test_simulator.py, test_feedback_available_exactly_at_decision_updates_before_selection: ispezionato, non rieseguito. |
| Warm-up | prepare_warmup() (320) genera contesti cronologici per tutti gli utenti e chiama scaler.fit, senza reward o feedback. prepare_warmup_from_precomputed() (383) importa parametri e riproduce eventi fino allo stesso confine. |
| Learner | src/recommender/policies.py, LinUCBPolicy.reset() (281) crea un nuovo learner: A=I15 e b=0, stessi alpha e dimensione. Nelle 15 feature non esiste un'intercetta esplicita. |
| Confini | run_phase() richiede coda vuota all'avvio e applica i feedback residui fino alla fine inclusiva. Il vincolo t+7d<=fine esclude finestre incomplete. |
| Test indipendente | src/experiments/runner.py, _run_one() (1334) crea nuove istanze di stato, generatore, contesto, scaler e policy. Il test osserva gli ascolti di validation precedenti alle proprie decisioni senza importare la memoria appresa o i feedback della validation. |
| Eleggibilità | _compute_eligibility() (415): almeno 10 eventi nel warm-up, 10 session start utilizzabili in validation e 10 in test; futuro completo e candidati non vuoti a tutte le profondità confrontate. La coorte usa quindi l'attività test anche per definire la popolazione della validation. |
| Personal | _top_personal_track_keys() (307): primi N item per chiave fra le tracce osservate degli artisti conosciuti, senza quote fra item già ascoltati e non. |
| Popular/Discovery | CandidateGenerator.generate() (355): popolarità item decrescente/chiave crescente per Popular; _top_discovery_track_keys() (342): primi N item non ascoltati per chiave. Familiar/new sono etichette, senza bilanciamento. |
| Metriche | src/evaluation/metrics.py, OfflineMetricsEvaluator.evaluate() (119): prima Top-K track, poi artisti distinti in prima occorrenza; Precision divide per m; IDCG usa min(K, numero artisti futuri rilevanti). Novelty track resta per item; novelty artista deduplicata; ILD su tutte le posizioni track; coverage globale di fase. |

I sei moduli scientifici state, simulator, candidate_generator, reward,
linucb e metrics non presentano differenze rispetto al commit ufficiale
della validation f67c7d9bdf2dfa24de0fe14221adb41c6f1e33a8.

## Antecedenza della scelta N=100

Il comando git show -s --format=fuller sul commit
c7167312f55fdacd86e4183bfd80a3d1b02d5b55 conferma data
7 ottobre 2026, 01:05:17 +0200.
Il README del runner in quella revisione, righe 38–56, dichiara N=100
principale a priori, N=50/200 di sensibilità, selezione NDCG@10 poi reward
poi parametro minore ed esclusione delle righe test dal selettore.
Config e runner della stessa revisione applicano i vincoli.
Il log persistente è
experiments/results/logs/validation-f67c7d9-20261019T153946Z.log;
la prima riga completata nel CSV ufficiale è datata
2026-10-19T18:44:04.459386+00:00.
È documentazione antecedente ai risultati, non preregistrazione formale.

## Interventi sulla relazione

- Capitolo 2: ipotesi UCB1, azioni non osservate, modello condiviso senza
  intercetta e conseguenze; nessuna nuova garanzia di regret.
- Capitolo 3: macro-media come scelta di peso fra utenti; attività ingrandita;
  concentrazione ridisegnata dai cinque punti già salvati nell'EDA.
- Capitolo 4: ordine feedback/stato/decisioni e caso di uguaglianza;
  estrazione Personal/Discovery senza quote di novelty.
- Capitolo 5: coorte retrospettiva, ritorno osservato, reset e coda,
  antecedenza N=100, esempio didattico con artisti duplicati e distinzione
  fra convenzioni implementate e letteratura.
- Capitolo 6: eliminato «preregistrata», paragrafo Global Popularity
  sulla stessa pagina, sweep più leggibili, figura Top-10 U04 dalla cache.
- Capitolo 7: bootstrap per cluster, IC bootstrap al 95% con scale separate
  per reward, NDCG e composizione; valori invariati.
- Capitolo 8: etichette e legenda dei trade-off più leggibili.
- Appendice: commit completo e fonte della predefinizione dello scenario.
- Bibliografia: tre nuove voci autentiche, nove citazioni complessive.

## Fonti bibliografiche controllate

- Ziegler, McNee, Konstan e Lausen (2005), WWW, pp. 22–32,
  DOI 10.1145/1060745.1060754. Fonte degli autori:
  <https://files.grouplens.org/papers/ziegler-www05.pdf>.
- Vargas e Castells (2011), RecSys, pp. 109–116,
  DOI 10.1145/2043932.2043955. Testo nel repository dell'università:
  <https://repositorio.uam.es/bitstreams/78fa337b-af26-4990-b56c-c1e8235afbd1/download>.
- Field e Welsh (2007), JRSS B 69(3), pp. 369–390,
  DOI 10.1111/j.1467-9868.2007.00593.x. Scheda dell'editore:
  <https://academic.oup.com/jrsssb/article-abstract/69/3/369/7109361>.
- Dataset: README originale Last.fm 1K, versione 1.0, marzo 2010,
  consultato localmente. Fonte specifica storica:
  <http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html>.
  Il sito non era raggiungibile alla verifica, come dichiarato in bibliografia.

Le citazioni nuove motivano concetti generali; non attribuiscono alla
letteratura le formule particolari di novelty/ILD del progetto.

## Verifiche e limiti

build_assets.py ha verificato CSV, manifesto, hash logici, 36+5 marker,
cutoff espliciti, 65 equivalenze metriche salvate e cinque effetti appaiati.
Tutte le tabelle numeriche preesistenti restano identiche byte per byte,
inclusi griglia delle 36 run e intervalli bootstrap.
Non sono stati ricalcolati bootstrap, policy o rilevanze dal TSV.

La figura U04 legge 30 record della cache canonica, il cui SHA-256 rimane
ed582fc9a26189785bf440be58b0a4ee9a419c77e20c8131bf0ea6ed7840341b.
La sola novelty locale rappresentata è quella artista; non si approssima
la track novelty dei singoli esempi.

Restano i limiti dichiarati: coorte retrospettiva, assenza di esposizioni
e di reward causale, finestra di sette giorni, troncamento lessicografico,
modello lineare condiviso, inferenza post-hoc condizionata alla traiettoria
e dipendenze fra utenti non modellate. L'uguaglianza dei timestamp richiede
la convenzione descritta, non la pretesa che tutta la memoria del learner
provenga da D_<t.

Compilazione e integrità sono registrate in data/quality_checks.json e
data/revision_checks.json. Immagini delle pagine e fotografia iniziale
dei file protetti restano in build/, ignorata da Git.
Nessun commit o push è stato eseguito.

## Integrazione argomentativa successiva

La versione corrente conserva gli otto capitoli e l'appendice e sviluppa
il percorso problema, dati, requisiti, architettura, protocollo e verifica
degli obiettivi. Non aggiunge algoritmi, ablation o nuove analisi dei log.

Nel Capitolo 1 il CMAB è motivato dalla decisione ripetuta con contesto e
catalogo variabili; il ritardo deriva dal proxy che accumula ascolti.
È distinta l'eventuale acquisizione di nuove risposte online dalla sequenza
offline invariata. Nel Capitolo 2 è chiarito il ruolo sperimentale di ogni
baseline e il fatto che LinUCB cambi anche rappresentazione e apprendimento.
La chiusura del Capitolo 3 traduce le osservazioni EDA in requisiti.
Il Capitolo 4 motiva separazione dei moduli, sorgenti candidate, famiglie di
feature e compromessi della reward. Il Capitolo 5 spiega split cronologici,
scenario principale, criterio di selezione e cutoff. I Capitoli 6–8
ricollegano accuratezza, scoperta e incertezza alla domanda iniziale.

### Natura delle motivazioni

- Evidenze osservate: eterogeneità, concentrazione, ripetizione e ID mancanti
  derivano dagli output EDA già citati; gap e riascolti sono esplorazioni
  preliminari con il vecchio parser, mantenute esplicitamente come tali.
- Scelte antecedenti documentate: N=100 principale, N=50/200 di sensibilità,
  NDCG@10, spareggi e cutoff figurano nel README e nel codice del commit
  c716731 del 7 ottobre; la selezione è confermata dal manifesto.
- Giustificazioni tecniche ricostruite: separare osservazioni e reward
  facilita la verifica causale; condividere il pool rende confrontabili le
  opportunità; il modello contestuale generalizza tramite le feature;
  il logaritmo attenua i conteggi elevati. Non sono presentate come esiti
  di confronti architetturali eseguiti.
- N=100 limita matematicamente la union a 300 item, rispetto a 150/600:
  questa è una giustificazione tecnica della dimensione intermedia,
  non la prova di un benchmark preliminare o dell'intenzione storica
  specifica che avrebbe determinato il valore.
- Restano operative o euristiche la profondità principale, le soglie
  temporali e di attività e la definizione del proxy artist-level.
  La soglia di dieci non è una necessità matematica del reward.
  Sette giorni non sono descritti come ottimali; la scelta dell'artista
  riconosce ascolti di altri titoli ma perde precisione sulla traccia.

Il costo dell'esplorazione è interpretato soltanto nel proxy offline;
non si deduce una perdita causale di soddisfazione. Sono conservati
convenzione del feedback simultaneo, coorte retrospettiva e limiti
dell'inferenza, senza equiparare vicinanza numerica ed equivalenza.

La versione argomentativa ha 27 pagine. Formule visualizzate, tabelle,
figure e valori decimali preesistenti sono invariati. Il confronto
SHA-256 conferma 172 file protetti immutati, inclusi codice, test,
notebook, configurazioni, manifesto, aggregati, marker, tabelle e figure.
Per i JSONL grandi resta il controllo dei metadati, non un nuovo rehash.
La verifica corrente è in data/argumentation_checks.json; il registro
data/revision_checks.json conserva la precedente revisione metodologica
di 25 pagine.
