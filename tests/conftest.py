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
            "label": "artificial intelligence",
            "category": "technology",
            "description": "Simulation of human intelligence",
            "content_type": "user",
            "source": "input"
        },
        {
            "label": "machine learning",
            "category": "technology",
            "description": "Automatic learning from data",
            "content_type": "user",
            "source": "input"
        },
        {
            "label": "neural networks",
            "category": "technology",
            "description": "Computational models inspired by the brain",
            "content_type": "assistant",
            "source": "output"
        }
    ]

@pytest.fixture
def sample_texts():
    """Testi di esempio per i test"""
    return {
        "input": "Artificial intelligence and machine learning are revolutionizing the world.",
        "output": "Neural networks are computational models that mimic the functioning of the human brain."
    } 