# /tests/test_mock_sentence_transformer.py
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import ConceptTransformationVisualizer

class TestMockSentenceTransformer(unittest.TestCase):
    
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
        
    def test_compute_embeddings_with_mock(self):
        """Test that compute_embeddings works with mocked encoder."""
        # Test concepts
        test_concepts = [
            {
                "label": "short",  # Length 5
                "category": "test",
                "description": "a very long description"  # Length 23
            },
            {
                "label": "longerterm",  # Length 10
                "category": "test",
                "description": "short desc"  # Length 10
            }
        ]
        
        # Compute embeddings
        result = self.visualizer.compute_embeddings(test_concepts, name_weight=0.7)
        
        # Check that embeddings were added
        self.assertIn("embedding", result[0])
        self.assertIn("embedding", result[1])
        
        # Verify that the mocked embeddings were used
        # For "short" with name_weight=0.7:
        # 0.7 * [1, 1, 1] + 0.3 * [2, 2, 2] = [1.3, 1.3, 1.3]
        # After normalization: [1.3, 1.3, 1.3] / sqrt(1.3^2 + 1.3^2 + 1.3^2)
        expected_embedding = 1.3 / np.sqrt(3 * 1.3**2)
        self.assertAlmostEqual(result[0]["embedding"][0], expected_embedding, places=5)
        
        # For "longerterm" with name_weight=0.7:
        # 0.7 * [1, 1, 1] + 0.3 * [2, 2, 2] = [1.3, 1.3, 1.3]
        # After normalization: [1.3, 1.3, 1.3] / sqrt(1.3^2 + 1.3^2 + 1.3^2)
        self.assertAlmostEqual(result[1]["embedding"][0], expected_embedding, places=5)
        
    def test_embedding_storage(self):
        """Test that embeddings are stored in the appropriate dictionaries."""
        # Test concept
        test_concept = {
            "label": "testconcept",
            "category": "test",
            "description": "test description"
        }
        
        # Compute embedding
        self.visualizer.compute_embeddings([test_concept])
        
        # Check storage in dictionaries
        self.assertIn("testconcept", self.visualizer.name_embeddings)
        self.assertIn("testconcept", self.visualizer.desc_embeddings)
        self.assertIn("testconcept", self.visualizer.concept_embeddings["names"])
        self.assertIn("testconcept", self.visualizer.concept_embeddings["descriptions"])
        
        # Check that the embeddings match what's expected from the mock
        # After normalization: [1, 1, 1] / sqrt(1^2 + 1^2 + 1^2) = 1/sqrt(3)
        expected_name = 1.0 / np.sqrt(3)
        self.assertAlmostEqual(self.visualizer.name_embeddings["testconcept"][0], expected_name, places=5)
        
        # After normalization: [2, 2, 2] / sqrt(2^2 + 2^2 + 2^2) = 2/sqrt(12)
        expected_desc = 2.0 / np.sqrt(12)
        self.assertAlmostEqual(self.visualizer.desc_embeddings["testconcept"][0], expected_desc, places=5)

    def test_embed_text(self):
        """Test embedding text."""
        text = "Test text"
        embedding = self.mock_encoder.embed_text(text)
        
        # Check embedding
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (384,))
        np.testing.assert_array_almost_equal(embedding, np.ones(384))
        
    def test_embed_texts(self):
        """Test embedding multiple texts."""
        texts = ["Test text 1", "Test text 2"]
        embeddings = self.mock_encoder.embed_texts(texts)
        
        # Check embeddings
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (2, 384))
        np.testing.assert_array_almost_equal(embeddings, np.ones((2, 384)))

if __name__ == '__main__':
    unittest.main()