# Riproducibilità dei notebook di analisi

Tutti i notebook conservano gli output delle analisi già eseguite: un clone pulito può quindi consultare tabelle e figure incorporate. Dataset e risultati sperimentali sono esclusi dal repository; l'esecuzione completa richiede gli artefatti indicati qui sotto.

| Notebook | Cosa si può rieseguire da un clone pulito | Prerequisiti per rieseguire l'analisi completa |
|---|---|---|
| `01_validation_analysis.ipynb` | Consultare gli output già incorporati. | CSV aggregato ufficiale e marker dei checkpoint di validation (`experiments/results/`, ignorata da Git). |
| `02_final_test_analysis.ipynb` | La sola sezione qualitativa: eseguire le celle di caricamento/verifica, catalogo e grafici. Usa la cache tracciata `notebooks/data/qualitative_examples.json`; non richiede dataset né risultati locali. | Il resto del notebook richiede gli aggregati validation/test e i checkpoint ufficiali. La cache è verificata rispetto agli hash congelati di dataset, configurazione e sorgenti. |
| `03_statistical_analysis.ipynb` | Consultare gli output statistici già incorporati. | Aggregati, cinque checkpoint JSONL dettagliati e TSV Last.fm completo. La ricostruzione statistica percorre l'intero TSV e crea un SQLite temporaneo; non rieseguirla per la sola consultazione. |

## Cache qualitativa

`notebooks/data/qualitative_examples.json` e il sidecar `.sha256` sono la copia pubblica verificata dei cinque esempi U01–U05 e delle 150 raccomandazioni. La cella di caricamento controlla il digest del file e la provenienza fissata nel manifesto e nella configurazione versionati. Gli eventuali `runs.csv` e marker locali vengono usati solo per controlli aggiuntivi e non sono prerequisiti per il rendering. Gli identificativi sorgente sono ID pseudonimi `user_…` del dataset; nelle tabelle e nei grafici vengono mostrati solo gli alias U01–U05. Non sono inclusi nomi reali, email o contatti.

Per rigenerare i cinque grafici dalla cache, aprire il notebook dalla root del repository ed eseguire le celle della sezione qualitativa. La visualizzazione include novelty a livello artista usando i flag pre-decisione verificati. La track novelty aggregata è nei risultati ufficiali, ma non è ricostruita per questi cinque esempi perché la storia esatta per item non è persistita. La reward Top-1 è un proxy offline, non una stima causale.

`notebooks/build_qualitative_cache.py` è uno strumento per manutentori: rigenera una cache di lavoro da CSV, marker e JSONL ufficiali quando tali artefatti sono disponibili. Non legge il TSV e non riseleziona i decision point; scrive in `.analysis-cache/`, ignorata da Git. Confrontare e verificare l'artefatto prima di sostituire la copia versionata.
