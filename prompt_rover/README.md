---
title: Prompt Rover
emoji: 🚀
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: "4.21.0"
app_file: app.py
pinned: false
license: mit
short_description: "Explore semantic relationships between text concepts"
---

# Prompt Rover - HuggingFace Spaces Deployment

> ⚠️ **Nota**: Questo README è specifico per il deployment su HuggingFace Spaces. 
> Per la **documentazione completa** del progetto, diagrammi di flusso, esempi dettagliati e istruzioni di sviluppo, consulta il [README principale](../README.md) nella root del repository.

## Quick Start

Questo Space permette di esplorare le relazioni semantiche tra concetti estratti dai testi utilizzando tecniche di embedding e visualizzazione interattiva.

### Come Usare:

1. **Modalità Input/Output**: Inserisci un testo di input e uno di output per confrontare i concetti
2. **Modalità Chat**: Conversa con il sistema e osserva l'evoluzione dei concetti

### Caratteristiche:
- Estrazione automatica di concetti con spaCy o OpenAI
- Visualizzazione interattiva con UMAP, t-SNE, PCA
- Grafi concettuali basati sulla similarità
- Supporto per evoluzione temporale in modalità chat

## Configurazione Opzionale

Per funzionalità avanzate, puoi fornire una **chiave API OpenAI** nelle opzioni avanzate per un'estrazione più sofisticata dei concetti.

## Architettura

Il progetto utilizza un'architettura modulare:
- `core/`: Logica principale (estrazione, embeddings, grafi)
- `ui/`: Interfaccia Gradio
- `chat/`: Gestione conversazioni
- `visualization/`: Visualizzazioni interattive e statiche
- `utils/`: Utilità comuni

## Link Utili

- 📖 [Documentazione completa](../README.md)
- 🐛 [Segnala problemi](https://github.com/your-username/prompt-rover/issues)
- 💡 [Contribuisci](https://github.com/your-username/prompt-rover/pulls)

---
*Powered by Gradio, SentenceTransformers, e NetworkX*

<!-- Deployment trigger: force new workflow run --> 