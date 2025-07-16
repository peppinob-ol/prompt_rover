"""
Configurazione e costanti per Prompt Rover
"""

import os
from pathlib import Path

# Percorsi
PROJECT_ROOT = Path(__file__).parent.parent
LOG_FILE = PROJECT_ROOT / "concept_transform_detailed.log"

# Configurazione modelli
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LLM_MODEL = "gpt-4"

# Schema colori
TEAL = "#3bb7b6"    # Colore user/input
ORANGE = "#fbad52"  # Colore assistant/output

# Nomi colonne standardizzati
CONTENT_TYPE = "content_type"  # Valori: "user", "assistant" 
MESSAGE_ID = "message_id"      # Ordinamento messaggi
LABEL = "label"                # Label del concetto
CATEGORY = "category"          # Categoria del concetto
DESCRIPTION = "description"    # Descrizione del concetto
X_COORD = "x"                  # Coordinata X per visualizzazione
Y_COORD = "y"                  # Coordinata Y per visualizzazione
ALPHA = "alpha"                # Valore trasparenza
EMBEDDING = "embedding"        # Embedding del concetto

# Configurazione cache
CACHE_ENABLED = True
MAX_CACHE_SIZE = 1000

# Limiti
MAX_CONCEPTS_PER_TEXT = 10
MAX_CHAT_CONCEPTS = 500
CONCEPT_MIN_LENGTH = 3

# Configurazione visualizzazione
DEFAULT_FIGURE_SIZE = (12, 10)
DEFAULT_SCATTER_SIZE = 100
DEFAULT_ALPHA = 0.7

# Configurazione riduzione dimensionale
UMAP_DEFAULT_NEIGHBORS = 5
UMAP_DEFAULT_MIN_DIST = 0.1
TSNE_DEFAULT_PERPLEXITY = 30

# Configurazione grafo
DEFAULT_K_NEIGHBORS = 3
DEFAULT_USE_MST = True
DEFAULT_SHOW_IO_LINKS = False

# Timeout API
LLM_API_TIMEOUT = 30

# Configurazione logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# Carica variabili d'ambiente
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Configurazione Gradio
GRADIO_THEME = "default"
GRADIO_SHARE = False
GRADIO_DEBUG = True 