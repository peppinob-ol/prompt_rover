# /tests/test_embedding.py
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import ConceptTransformationVisualizer

class TestEmbedding(unittest.TestCase):
    
    @patch('main.SentenceTransformer')
    def setUp(self, mock_sentence_transformer):
        """Set up the test environment."""
        # Create a mock encoder that returns predictable embeddings
        self.mock_encoder = MagicMock()
        # For labels we'll return all ones, for descriptions all twos
        self.mock_encoder.encode.side_effect = lambda texts: np.ones((len(texts), 3)) if any("concept" in t for t in texts) else np.ones((len(texts), 3)) * 2
        mock_sentence_transformer.return_value = self.mock_encoder
        
        # Initialize the visualizer
        self.visualizer = ConceptTransformationVisualizer(embedding_model="test-model")
        
        # Test concepts
        self.test_concepts = [
            {
                "label": "test concept 1",
                "category": "entity",
                "description": "description 1"
            },
            {
                "label": "test concept 2",
                "category": "process",
                "description": "description 2"
            }
        ]
        
    def test_compute_embeddings(self):
        """Test the compute_embeddings method."""
        # Compute embeddings
        concepts_with_embeddings = self.visualizer.compute_embeddings(self.test_concepts, name_weight=0.7)
        
        # Check that the encoder was called twice (once for labels, once for descriptions)
        self.assertEqual(self.mock_encoder.encode.call_count, 2)
        
        # Check that embeddings were added to the concepts
        for concept in concepts_with_embeddings:
            self.assertIn("embedding", concept)
            self.assertEqual(concept["embedding"].shape, (3,))
            
            # Check that the embedding is normalized
            self.assertAlmostEqual(np.linalg.norm(concept["embedding"]), 1.0, places=5)
            
            # Check that the embedding is a weighted combination
            # For name_weight=0.7, it should be 0.7*1 + 0.3*2 = 1.3, then normalized
            expected_value = 0.7 * 1 + 0.3 * 2
            expected_value = expected_value / np.sqrt(3 * expected_value**2)  # Normalize
            self.assertAlmostEqual(concept["embedding"][0], expected_value, places=5)
            
        # Check that embeddings were stored in the class dictionaries
        self.assertEqual(len(self.visualizer.name_embeddings), 2)
        self.assertEqual(len(self.visualizer.desc_embeddings), 2)
        self.assertIn("test concept 1", self.visualizer.name_embeddings)
        self.assertIn("test concept 1", self.visualizer.desc_embeddings)
        
        # Check that embeddings were also stored in the concept_embeddings structure
        self.assertEqual(len(self.visualizer.concept_embeddings['names']), 2)
        self.assertEqual(len(self.visualizer.concept_embeddings['descriptions']), 2)
        
    def test_compute_embeddings_empty(self):
        """Test handling of empty concept list."""
        # Compute embeddings for an empty list
        result = self.visualizer.compute_embeddings([])
        
        # Check that the result is an empty list
        self.assertEqual(result, [])
        
        # Check that the encoder was not called
        self.mock_encoder.encode.assert_not_called()

if __name__ == '__main__':
    unittest.main()