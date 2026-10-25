# Relazione Music Discovery CMAB

Relazione completa in italiano, con sorgenti modulari, figure e tabelle già
generate. Il PDF consegnabile è music_discovery_cmab.pdf. I dati personali
del frontespizio si configurano nelle cinque macro all'inizio di main.tex.

## Compilazione senza dataset

Per compilare bastano main.tex, sections/, figures/, tables/ e references.bib:
non occorrono dataset, SQLite, checkpoint o Python. Su Overleaf caricare
questi file mantenendo la struttura e impostare main.tex come documento
principale, con compilatore pdfLaTeX e bibliografia BibTeX.

Da un terminale con TeX Live o MiKTeX, nella directory report:

    latexmk -pdf main.tex

In alternativa:

    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex

La compilazione locale di consegna usa build/ per i file ausiliari.
Con questa opzione creare anche build/sections/, eseguire il primo pdfLaTeX
con -output-directory=build, poi BibTeX da build/ con BIBINPUTS=..; e infine
due ulteriori passaggi pdfLaTeX dalla directory report. Non eseguire script
che rigenerano i notebook per compilare la relazione.

## Aggiornamento controllato di figure e tabelle

Dalla root del repository:

    .\.venv\Scripts\python.exe report/build_assets.py

Lo script legge i due CSV ufficiali, i 41 marker, gli output incorporati
nei notebook EDA e statistico e la cache qualitativa tracciata.
Richiede pandas, matplotlib e numpy; usa la libreria standard per leggere
gli output HTML del notebook. Non esegue celle, non legge il TSV di ascolti,
non ricalcola bootstrap, non esegue policy.

Le tabelle sono arrotondate a sei decimali; i CSV derivati conservano i valori
numerici degli aggregati. data/provenance.json conserva commit corrente,
hash delle fonti, run ID e checksum dichiarati dei dettagli.
data/statistical_tables.json conserva le tabelle statistiche estratte.
I JSONL dettagliati non sono ricalcolati né ricaricati in memoria.

## Note metodologiche emerse dal codice

- Il contesto ha 15 componenti: 13 standardizzate e due flag binari.
- Il warm-up seleziona i contesti fra le date ufficiali, ma lo stato conserva
  anche gli eventi del file anteriori all'inizio del campione.
- L'eleggibilità della coorte richiede attività utilizzabile in validation
  e test; la relazione ne esplicita il condizionamento.
- Gli output EDA precedono lo schema canonico finale; le cardinalità EDA
  non diventano denominatori della coverage.
- Il notebook preliminare sulle sessioni usa il vecchio parsing e riporta
  19.098.853 righe e una coorte diversa. È citato solo come analisi esplorativa.
- La cache qualitativa conserva artist novelty esatta; la track novelty dei
  singoli esempi non è disponibile e non viene approssimata.
- Gli intervalli statistici sono quelli già verificati nel notebook,
  post-hoc e condizionati alla traiettoria appresa.

## Revisione prima della consegna

Il frontespizio contiene i dati forniti dall’autore. La revisione finale è
documentata in REVIEW.md: 27 pagine, con raccordi progettuali, chiarimenti metodologici e una
figura qualitativa reale. Le interpretazioni dei risultati
e i limiti dell'inferenza sono formulati per una valutazione descrittiva
offline, senza vantaggi causali o nuove selezioni sul test.

Il registro delle verifiche è in CHECKS.md; data/argumentation_checks.json
registra i controlli della versione corrente. data/revision_checks.json
conserva la precedente revisione metodologica di 25 pagine. Le figure ridisegnate
leggono soltanto CSV e output salvati; la figura qualitativa legge la cache
versionata. Nessuna scansione del TSV è richiesta.
