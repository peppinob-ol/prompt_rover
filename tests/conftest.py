"""
Configurazione pytest per i test di Prompt Rover
"""

import sys
import os
import pytest

# Aggiungi il percorso del progetto al path Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disabilita logging durante i test per output più pulito
import logging
logging.disable(logging.CRITICAL)

@pytest.fixture
def mock_openai_key():
    """Mock della chiave OpenAI per i test"""
    return "test-key-12345"

@pytest.fixture
def sample_concepts():
    """Concetti di esempio per i test"""
    return [
        {
            "label": "intelligenza artificiale",
            "category": "tecnologia",
            "description": "Simulazione dell'intelligenza umana",
            "content_type": "user",
            "source": "input"
        },
        {
            "label": "machine learning",
            "category": "tecnologia",
            "description": "Apprendimento automatico dai dati",
            "content_type": "user",
            "source": "input"
        },
        {
            "label": "reti neurali",
            "category": "tecnologia",
            "description": "Modelli computazionali ispirati al cervello",
            "content_type": "assistant",
            "source": "output"
        }
    ]

@pytest.fixture
def sample_texts():
    """Testi di esempio per i test"""
    return {
        "input": "L'intelligenza artificiale e il machine learning stanno rivoluzionando il mondo.",
        "output": "Le reti neurali sono modelli computazionali che imitano il funzionamento del cervello umano."
    } 