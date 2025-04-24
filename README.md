# Prompt Rover - tracing LLM Latent Space

## Panoramica
Il `ConceptTransformationVisualizer` è uno strumento per estrarre, analizzare e visualizzare concetti da testi in modo interattivo. Utilizzando tecniche di embedding, riduzione dimensionale e visualizzazione di grafi, permette di esplorare le relazioni semantiche tra concetti in diversi contesti testuali.

## Funzionalità principali

- **Estrazione di concetti**: Estrazione automatica di concetti chiave da testi utilizzando modelli linguistici (spaCy o OpenAI)
- **Embedding semantico**: Conversione di concetti in vettori semantici utilizzando SentenceTransformer
- **Visualizzazione interattiva**: Rappresentazione bidimensionale dei concetti con UMAP, t-SNE o PCA
- **Analisi grafica**: Costruzione e visualizzazione di grafi concettuali basati su similarità
- **Modalità Input/Output**: Confronto tra concetti in testi di input e output
- **Modalità Chat**: Visualizzazione dell'evoluzione dei concetti durante una conversazione

## Requisiti

- Python 3.7+
- Dipendenze principali:
  ```
  sentence-transformers
  spacy
  networkx
  umap-learn
  matplotlib
  pandas
  gradio
  ```
- Modelli spaCy (utilizzare `python -m spacy download it_core_news_sm` o equivalente)
- Opzionale: Chiave API OpenAI per estrazione concetti avanzata

## Utilizzo

### Tramite interfaccia Gradio

1. Eseguire `run_in_colab()` in Google Colab o lanciare lo script principale
2. Utilizzare l'interfaccia per:
   - **Modalità Input/Output**: Inserire testi di input e output per comparare concetti
   - **Modalità Chat**: Conversare e vedere l'evoluzione dei concetti nei messaggi

### Tramite API Python

```python
from concept_transform import ConceptTransformationVisualizer

# Inizializzazione
visualizer = ConceptTransformationVisualizer(
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    openai_api_key="your_api_key"  # Opzionale
)

# Modalità Input/Output
df, fig, status = visualizer.process_text_pair(
    input_text="testo di input",
    output_text="testo di output",
    use_llm=True,  # Usa OpenAI per estrazione concetti
    dim_reduction="umap"  # o "tsne", "pca"
)

# Visualizzazione personalizzata
fig = visualizer.visualize_concepts(
    df,
    show_evolution=False,
    title="Mia visualizzazione personalizzata"
)
```

# Diagramma di Flusso: ConceptTransformationVisualizer

```mermaid
flowchart TD
    A[Input Testo/Conversazione] --> B[Estrazione Concetti]
    B -->|LLM| B1[extract_concepts_with_llm]
    B -->|Alternativo| B2[extract_concepts_alternative]
    B1 & B2 --> C[Calcolo Embedding]
    
    C -->|SentenceTransformer| D[Analisi Rete]
    
    D -->|MST o k-NN| D1[build_concept_graph]
    D1 --> E[Riduzione Dimensionale]
    
    E -->|UMAP| E1[reduce_dimensions]
    E -->|t-SNE| E2[reduce_dimensions]
    E -->|PCA| E3[reduce_dimensions]
    
    E1 & E2 & E3 --> F[Visualizzazione]
    
    F -->|Standard| F1[visualize_concepts\nshow_evolution=false]
    F -->|Evoluzione| F2[visualize_concepts\nshow_evolution=true]
    
    F1 & F2 --> G[Output Grafico]
    
    H[(Cache)] <-.-> B
    H <-.-> C
    H <-.-> D
    H <-.-> E
    
    subgraph "Modalità Input/Output"
    I[process_text_pair]
    end
    
    subgraph "Modalità Chat"
    J[process_new_message]
    end
    
    I --> A
    J --> A
    
    style F1 fill:#TEAL,color:white
    style F2 fill:#ORANGE,color:white
    style I fill:#d0e0ff,stroke:#333
    style J fill:#ffead0,stroke:#333
```

## Descrizione del flusso

1. **Input**: Il processo inizia con un testo o messaggio di conversazione
   - In modalità Input/Output: elaborazione di due testi separati
   - In modalità Chat: elaborazione di messaggi sequenziali

2. **Estrazione Concetti**: Identificazione di concetti chiave dal testo
   - Metodo LLM (`extract_concepts_with_llm`): Utilizza OpenAI per estrazione avanzata
   - Metodo Alternativo (`extract_concepts_alternative`): Utilizza spaCy per estrazione locale

3. **Calcolo Embedding**: Trasformazione dei concetti in vettori semantici
   - `compute_embeddings`: Elabora sia il nome che la descrizione di ogni concetto

4. **Analisi Rete**: Costruzione di un grafo basato su relazioni tra concetti
   - `build_concept_graph`: Crea un MST (Minimum Spanning Tree) o grafo k-NN

5. **Riduzione Dimensionale**: Proiezione degli embedding in spazio 2D
   - `reduce_dimensions`: Implementa UMAP, t-SNE o PCA in base alla preferenza

6. **Visualizzazione**: Creazione del grafico finale
   - `visualize_concepts`: Metodo unificato con supporto per evoluzione temporale
   - Output standard o con evoluzione temporale (per modalità Chat)

7. **Cache**: Memorizzazione di risultati intermedi per ottimizzare performance
   - Riutilizzo di concetti, embedding, grafi e proiezioni quando possibile

Questo flusso si applica sia alla modalità Input/Output (tramite `process_text_pair`) che alla modalità Chat (tramite `process_new_message`), con differenze nella gestione dello stato tra messaggi sequenziali.


## Struttura del codice

- **Estrazione concetti**: `extract_concepts_with_llm()` e `extract_concepts_alternative()`
- **Gestione embedding**: `compute_embeddings()`
- **Costruzione grafo**: `build_concept_graph()`
- **Riduzione dimensionale**: `reduce_dimensions()`
- **Visualizzazione**: `visualize_concepts()` (unificata)
- **Flussi di processo**: `process_text_pair()` e `process_new_message()`
- **Interfaccia**: Funzioni per l'integrazione con Gradio

## Miglioramenti recenti

1. **Risposta di chat basata sulla conversazione**:
   - Genera risposte contestuali anziché statiche nella modalità chat
   - Fallback a risposte variabili predefinite se OpenAI non è disponibile

2. **Gestione errori migliorata**:
   - Gestione timeout nelle chiamate API
   - Fallback automatico a metodi alternativi
   - Logging dettagliato degli errori

3. **Inizializzazione OpenAI centralizzata**:
   - Metodo unificato `initialize_openai_client()`
   - Consistenza nella gestione delle chiavi API

4. **Sistema di caching coerente**:
   - Cache per concetti, embedding, riduzioni dimensionali e grafi
   - Riutilizzo dei reducer dimensionali quando possibile
   - Chiavi di cache parametrizzate

5. **Visualizzazione unificata**:
   - Metodo `visualize_concepts()` che sostituisce i precedenti metodi separati
   - Costanti standardizzate per colori e nomi di colonne
   - Gestione robusta dei nomi delle colonne per retrocompatibilità

## Test manuali consigliati

Per testare la visualizzazione, suggeriamo:

1. **Test base Input/Output**:
   - Utilizzare l'esempio precaricato "fiori e saggezza"
   - Verificare che i concetti di input e output siano distinguibili
   - Controllare che i collegamenti del grafo siano appropriati

2. **Test modalità Chat**:
   - Inviare 3-4 messaggi correlati (es. su un argomento specifico)
   - Verificare che i colori distinguano utente e sistema
   - Controllare che concetti simili siano collegati tra messaggi
   - Verificare che l'evoluzione temporale sia visibile (concetti più vecchi più trasparenti)

3. **Test di visualizzazione con parametri diversi**:
   - Provare diverse riduzioni dimensionali (UMAP, t-SNE, PCA)
   - Alternare tra MST e k-NN per la costruzione del grafo
   - Modificare il peso nome/descrizione e verificare l'impatto

4. **Test di resilienza**:
   - Provare testi molto brevi o molto lunghi
   - Verificare comportamento con testi in diverse lingue
   - Inserire caratteri speciali o emoji

## Sviluppi futuri

- Ulteriore refactoring per modularità
- Suite di test automatizzati
- Supporto multilingua migliorato
- Modalità di confronto di più testi
- Esportazione di grafi e risultati

## Note per i tester

Quando si verifica la visualizzazione, considerare:

1. **Leggibilità**: Etichette, colori e disposizione sono facilmente interpretabili?
2. **Coerenza semantica**: I concetti simili appaiono vicini nel grafico?
3. **Efficacia del grafo**: I collegamenti riflettono relazioni significative?
4. **Performance**: Il sistema risponde in tempi ragionevoli con testi di diverse dimensioni?
5. **Esperienza utente**: L'interfaccia e le visualizzazioni sono intuitive?

I feedback su questi aspetti saranno preziosi per miglioramenti futuri.