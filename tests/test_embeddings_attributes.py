# tests/test_embeddings_attributes.py

import unittest
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Aggiungi la directory principale al path di Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ora importa direttamente
from main import ConceptTransformationVisualizer

class TestEmbeddingsAttributes(unittest.TestCase):
    """Test per verificare la corretta inizializzazione e uso degli attributi degli embedding"""
    
    def setUp(self):
        """Inizializza un visualizzatore per i test"""
        print("Inizializzazione visualizzatore per il test...")
        self.visualizer = ConceptTransformationVisualizer()
        print("Visualizzatore inizializzato")
    
    def test_attributes_initialization(self):
        """Verifica che tutti gli attributi per gli embedding siano inizializzati correttamente"""
        print("Test inizializzazione attributi...")
        # Verifica che gli attributi esistano nell'istanza
        self.assertTrue(hasattr(self.visualizer, 'name_embeddings'), 
                        "L'attributo 'name_embeddings' non esiste")
        self.assertTrue(hasattr(self.visualizer, 'desc_embeddings') or 
                        hasattr(self.visualizer, 'description_embeddings'),
                        "Né 'desc_embeddings' né 'description_embeddings' esistono")
        self.assertTrue(hasattr(self.visualizer, 'concept_embeddings'),
                        "L'attributo 'concept_embeddings' non esiste")
        
        # Verifica che siano dizionari vuoti
        self.assertEqual(len(self.visualizer.name_embeddings), 0,
                        "name_embeddings non è un dizionario vuoto all'inizializzazione")
        self.assertEqual(len(self.visualizer.concept_embeddings['names']), 0,
                        "concept_embeddings['names'] non è un dizionario vuoto all'inizializzazione")
        print("Test inizializzazione attributi completato")
    
    # ... [altri test] ...

if __name__ == "__main__":
    unittest.main()