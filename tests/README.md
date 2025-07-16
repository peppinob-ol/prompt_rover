# Test Suite di Prompt Rover

## Panoramica

La suite di test di Prompt Rover è organizzata per validare tutti i componenti del sistema in modo modulare e completo.

## Struttura dei Test

### Test di Base
- `test_imports.py` - Verifica che tutti i moduli siano importabili correttamente
- `test_refactoring_validation.py` - Test completo di validazione del refactoring con mock delle dipendenze pesanti

### Test per Componente
- `test_concept_extractor.py` - Test per l'estrazione concetti (LLM e spaCy)
- `test_embedding_manager.py` - Test per la gestione embeddings
- `test_cache_manager.py` - Test per il sistema di caching LRU
- `test_integration.py` - Test di integrazione end-to-end

### Test Esistenti (da refactorizzare)
- Altri file di test che necessitano aggiornamento per la nuova struttura

## Esecuzione Test

### Test Rapido con Mock
```bash
python tests/test_refactoring_validation.py
```

### Test Completo
```bash
python tests/test_runner.py
```

### Test Specifico
```bash
python tests/test_runner.py tests/test_imports.py
```

## Dipendenze per i Test

### Dipendenze Base
- `pytest`
- `numpy`
- `pandas`
- `matplotlib`
- `networkx`
- `spacy` (con modello `en_core_web_sm`)

### Dipendenze Opzionali (mockate nei test)
- `sentence-transformers`
- `sklearn`
- `umap-learn`
- `plotly`
- `gradio`

## Copertura

I test coprono:
- ✅ Struttura del package
- ✅ Import di tutti i moduli
- ✅ Integrazione tra componenti core
- ✅ Visualizzatore principale (modalità input/output)
- ✅ Modalità chat
- ✅ Sistema di caching
- ✅ Gestione errori
- ✅ Configurazioni e costanti

## Note per Sviluppatori

1. **Mock delle Dipendenze**: Il file `test_refactoring_validation.py` utilizza mock per le dipendenze pesanti (sentence-transformers, sklearn, etc.) per velocizzare i test

2. **Test Isolati**: Ogni componente ha test isolati che verificano le funzionalità specifiche

3. **Test di Integrazione**: `test_integration.py` verifica che tutti i componenti funzionino insieme correttamente

4. **Fixtures**: Usa `conftest.py` per fixtures condivise come dati di esempio e configurazioni mock 