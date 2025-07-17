# /tests/test_caching.py
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import ConceptTransformationVisualizer

class TestCaching(unittest.TestCase):
    
    @patch('main.SentenceTransformer')
    def setUp(self, mock_sentence_transformer):
        """Set up the test environment."""
        # Create a mock for the encoder
        mock_encoder = MagicMock()
        mock_sentence_transformer.return_value = mock_encoder
        
        # Initialize the visualizer
        self.visualizer = ConceptTransformationVisualizer(embedding_model="test-model")
        
    def test_get_cache_key(self):
        """Test the _get_cache_key method."""
        # Generate a cache key for a text
        key1 = self.visualizer._get_cache_key("test text")
        
        # Check that the key is a string
        self.assertIsInstance(key1, str)
        
        # Generate a cache key with parameters
        key2 = self.visualizer._get_cache_key("test text", {"param1": "value1", "param2": 123})
        
        # Check that keys are different
        self.assertNotEqual(key1, key2)
        
        # Check that the same inputs produce the same key
        key3 = self.visualizer._get_cache_key("test text", {"param1": "value1", "param2": 123})
        self.assertEqual(key2, key3)
        
        # Check that different texts produce different keys
        key4 = self.visualizer._get_cache_key("different text", {"param1": "value1", "param2": 123})
        self.assertNotEqual(key2, key4)
        
    @patch.object(ConceptTransformationVisualizer, 'extract_concepts_alternative')
    @patch.object(ConceptTransformationVisualizer, 'compute_embeddings')
    def test_process_text_pair_caching(self, mock_compute_embeddings, mock_extract_concepts):
        """Test caching in process_text_pair method."""
        # Set up mocks
        mock_extract_concepts.return_value = [{"label": "concept1"}]
        mock_compute_embeddings.return_value = [{"label": "concept1", "embedding": np.array([0.1, 0.2, 0.3])}]
        
        # Process a text pair
        self.visualizer.process_text_pair("input text", "output text", use_llm=False)
        
        # Check that extraction and embedding were called
        self.assertEqual(mock_extract_concepts.call_count, 2)  # Once for input, once for output
        self.assertEqual(mock_compute_embeddings.call_count, 2)
        
        # Process the same text pair again
        self.visualizer.process_text_pair("input text", "output text", use_llm=False)
        
        # Check that extraction and embedding were not called again
        self.assertEqual(mock_extract_concepts.call_count, 2)  # Still just the original calls
        self.assertEqual(mock_compute_embeddings.call_count, 2)
        
        # Process a different text pair
        self.visualizer.process_text_pair("new input", "new output", use_llm=False)
        
        # Check that extraction and embedding were called for the new texts
        self.assertEqual(mock_extract_concepts.call_count, 4)  # Two more calls
        self.assertEqual(mock_compute_embeddings.call_count, 4)
        
    @patch.object(ConceptTransformationVisualizer, 'extract_concepts_alternative')
    @patch.object(ConceptTransformationVisualizer, 'compute_embeddings')
    def test_chat_caching(self, mock_compute_embeddings, mock_extract_concepts):
        """Test caching in chat mode."""
        # Set up mocks
        mock_extract_concepts.return_value = [{"label": "concept1"}]
        mock_compute_embeddings.return_value = [{"label": "concept1", "embedding": np.array([0.1, 0.2, 0.3])}]
        
        # Process a chat message
        self.visualizer.process_new_message("hello", is_user=True, use_llm=False)
        
        # Check that extraction and embedding were called
        mock_extract_concepts.assert_called_once()
        mock_compute_embeddings.assert_called_once()
        
        # Reset mocks
        mock_extract_concepts.reset_mock()
        mock_compute_embeddings.reset_mock()
        
        # Process the same message again
        self.visualizer.process_new_message("hello", is_user=True, use_llm=False)
        
        # Check that extraction and embedding were not called again
        mock_extract_concepts.assert_not_called()
        mock_compute_embeddings.assert_not_called()

if __name__ == '__main__':
    unittest.main()