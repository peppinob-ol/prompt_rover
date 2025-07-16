"""
Prompt Rover - Tracing LLM Latent Space through prompting

Un tool per estrarre, analizzare e visualizzare concetti dai testi in modo interattivo.
"""

__version__ = "0.1.0"
__author__ = "Prompt Rover Team"

# Imports principali per comodità
from .core.visualizer import ConceptTransformationVisualizer

__all__ = [
    "ConceptTransformationVisualizer",
    "__version__"
] 