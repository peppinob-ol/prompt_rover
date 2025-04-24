# test_direct.py - File di test diretto nella directory principale

import unittest
from main import ConceptTransformationVisualizer

class SimpleEmbeddingTest(unittest.TestCase):
    def test_attributes_exist(self):
        """Test semplice per verificare che gli attributi esistano"""
        print("Inizializzando visualizzatore...")
        visualizer = ConceptTransformationVisualizer()
        print("Visualizzatore inizializzato, verifico attributi")
        
        # Verifica che gli attributi esistano
        self.assertTrue(hasattr(visualizer, 'name_embeddings'), 
                        "name_embeddings non esiste!")
        self.assertTrue(hasattr(visualizer, 'desc_embeddings'), 
                        "desc_embeddings non esiste!")
        self.assertTrue(hasattr(visualizer, 'concept_embeddings'), 
                        "concept_embeddings non esiste!")
        
        # Verifica che siano dizionari
        self.assertEqual(type(visualizer.name_embeddings), dict,
                        "name_embeddings non è un dizionario")
        self.assertEqual(type(visualizer.concept_embeddings['names']), dict,
                        "concept_embeddings['names'] non è un dizionario")
        
        print("Test completato con successo!")

if __name__ == "__main__":
    unittest.main()