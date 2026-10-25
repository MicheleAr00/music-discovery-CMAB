# Registro delle verifiche

## Fonti e numeri

- Aggregati ufficiali: 36 righe uniche validation, cinque righe uniche test.
- Marker controllati: 36 + 5, tutti completed, identità coerenti con CSV e
  dataset/config/commit/scaler; ogni metrica del marker coincide con CSV
  entro 1e-12. File dettagliati presenti. I SHA-256 registrati sono conservati
  nel registro di provenienza; non è stato ripetuto il rehash multi-GB.
- Hash del CSV validation confrontato con il manifesto; hash del CSV test
  confrontato con l'atteso nel notebook statistico.
- Hash logici delle due configurazioni e hash byte del final_test.json
  coincidono con il manifesto. Nessun file sperimentale viene scritto.
- Alias coverage verificati come @20, non usati come @10.
- Tabelle principali, sensitivity e griglia completa generate dai valori
  ufficiali. Le differenze validation/test sono calcolate sui valori non
  arrotondati e mostrate a sei decimali.
- Statistica: 65 confronti ricostruzione/aggregato riportati come validi negli
  output già eseguiti del notebook; cinque differenze LinUCB meno Greedy
  coincidono con le differenze degli aggregati entro l'arrotondamento 5.1e-7.
- Cache qualitativa: SHA-256 ed582fc9a26189785bf440be58b0a4ee9a419c77e20c8131bf0ea6ed7840341b.
  Top-1, titoli, nomi, reward e conteggi artist novelty di U01/U04 sono
  letti direttamente dalla cache. Nessuna track novelty locale inventata.

## EDA

Statistiche tratte dagli output di 01_dataset_analysis: celle 15 (struttura),
18 (utenti e distribuzione attività), 29 (cardinalità esplorative), 39/40
(concentrazione), 44 (ripetizione), 49/50 (concentrazione individuale),
59 (durata), 62 (ID mancanti). La figura della cella 21 è estratta e
ingrandita; quella di concentrazione è ridisegnata dai cinque valori
salvati nelle celle 39 e 40, senza leggere il dataset.

reward_sessions_split è citato come esplorazione preliminare (celle 1/2),
esplicitandone parsing obsoleto e numerosità differente. Il suo conteggio
di sessioni non sostituisce l'eleggibilità ufficiale.

## Codice e formule

Verificati identity.py, state.py, candidate_generator.py, context.py,
scaler.py, policies.py, linucb.py, reward.py, simulator.py, dataset.py,
runner.py e configurazioni ufficiali. La relazione esplicita:
15 componenti, modello lineare condiviso, tredici standardizzazioni,
gap strettamente maggiore di 10 minuti, intervallo futuro (t,t+7d],
IDCG basata su min(K, numero artisti rilevanti), ILD su tutte le posizioni
track-level con distanza artista, fit su tutti gli utenti del warm-up,
e condizionamento della coorte sull'attività anche nel test.

## Compilazione e revisione

PDF compilato con pdfLaTeX e BibTeX locali. Le immagini delle pagine sono
renderizzate con Poppler per la verifica visiva; gli intermedi rimangono
nella directory ignorata build/. Il log finale è privo di errori,
riferimenti indefiniti e overfull/underfull box. Il PDF conclusivo ha 27
pagine complessive; contiene 31 label, 13 riferimenti incrociati distinti
usati e nove citazioni bibliografiche risolte. Le pagine sono state
controllate visivamente, comprese tabelle numeriche, grafici, formule
e frontespizio. Il riepilogo automatico è in data/quality_checks.json.

Nel frontespizio sono inseriti Michele Arietta, matricola 637929, docente
Giuseppe Sansonetti e anno accademico 2025/2026, come forniti dall'autore.
Sono inserite anche la laurea magistrale in Ingegneria Informatica
(curriculum Intelligenza Artificiale e Machine Learning) e la dicitura
Progetto per l'esame di Sistemi Intelligenti per Internet.
Rimane da approvare editorialmente il testo prima della consegna.
Non sono stati avviati training, validation, test o scansioni del TSV.

## Revisione critica finale

La verifica del 23 ottobre 2026 è documentata in REVIEW.md. Il feedback
maturo con scadenza <= t può includere un ascolto a t; stato e feature
restano basati su D_<t. Questa convenzione è ora esplicita nel Capitolo 4.
Sono chiariti reset A=I15, b=0, coorte retrospettiva, IDCG ed esempio
numerico, troncamento Discovery senza quote e antecedenza di N=100.
La figura qualitativa U04 mostra 30 raccomandazioni reali dalla cache.
Le 13 tabelle preesistenti e i dati statistici salvati sono identici byte
per byte; 139 input piccoli protetti sono invariati. Per 101 file JSONL
sono verificate dimensione e data di modifica, senza rehash multi-GB.
Nella revisione metodologica il paragrafo Global Popularity era intero
alla pagina fisica 16; nella versione argomentativa rimane intero alla
pagina fisica 18 (stampata 15).
Compilazione della revisione metodologica: 25 pagine, zero warning, nessun riferimento
mancante; tutte le pagine renderizzate e ispezionate. Sono verificati
indice, formule, bibliografia e leggibilità delle nuove figure.
I controlli automatici aggiornati sono in data/revision_checks.json.

## Verifica della versione argomentativa

Documento corrente: 27 pagine, otto capitoli e appendice invariati nella
struttura. Sono aggiunte motivazioni e transizioni, distinguendo dati EDA,
scelte precedenti agli esperimenti e giustificazioni tecniche ricostruite.
I limiti euristici sono dichiarati; nessun confronto inesistente è
presentato come prova empirica. Il dettaglio è in REVIEW.md.
Formule visualizzate, tabelle, figure e numeri decimali preesistenti sono
identici; 172 file protetti hanno SHA-256 invariato. Non sono stati
rigenerati grafici o statistiche. Nessuna policy o scansione è eseguita.
Compilazione corrente: zero warning, riferimenti e bibliografia risolti.
Tutte le pagine sono renderizzate e controllate; l’ultima impaginazione
delle pagine 3 e 11–15 è stata ispezionata nuovamente.
Il riepilogo corrente è in data/argumentation_checks.json.

## Rifiniture editoriali conclusive — 25 ottobre 2026

La Tabella 4.1 è ora in ordine di lettura prima della standardizzazione;
la prima citazione chiude la pagina stampata 7 e la tabella apre la 8.
La sezione 4.5 tratta il reward e il feedback differito; la 4.6 raccoglie
riuso del warm-up, replay indipendente, costi e risorse. Il relativo
paragrafo è stato spostato senza riscriverlo. Nel Capitolo 5 il conteggio
del warm-up è chiamato storico; N_train conserva il significato originale.
Prosa del Capitolo 4 invariata salvo titoli e ordine; formule, contenuto
delle tabelle, figure e numeri preesistenti identici. I 181 file protetti
sono invariati; la convenzione dei feedback simultanei è preservata.
Compilazione definitiva: 27 pagine, zero warning, indice aggiornato e
nessun riferimento mancante. Le sei pagine cambiate sono state controllate
visivamente; tutte le altre hanno rendering identico alla versione
approvata. Evidenze in data/editorial_checks.json. Nessun esperimento,
commit o push è stato eseguito.
