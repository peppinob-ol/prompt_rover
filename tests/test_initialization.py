# /tests/test_initialization.py
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import ConceptTransformationVisualizer

class TestInitialization(unittest.TestCase):
    
    @patch('main.SentenceTransformer')
    def test_initialization_without_openai(self, mock_sentence_transformer):
        """Test initialization without OpenAI API key."""
        # Setup the mock
        mock_instance = MagicMock()
        mock_sentence_transformer.return_value = mock_instance
        
        # Initialize the visualizer
        visualizer = ConceptTransformationVisualizer(embedding_model="test-model")
        
        # Assertions
        mock_sentence_transformer.assert_called_once_with("test-model")
        self.assertIsNone(visualizer.llm_client)
        self.assertIsInstance(visualizer.name_embeddings, dict)
        self.assertIsInstance(visualizer.desc_embeddings, dict)
        
    @patch('main.SentenceTransformer')
    @patch('openai.OpenAI')
    def test_initialization_with_openai(self, mock_openai, mock_sentence_transformer):
        """Test initialization with OpenAI API key."""
        # Setup the mocks
        mock_st_instance = MagicMock()
        mock_sentence_transformer.return_value = mock_st_instance
        
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        # Initialize the visualizer
        visualizer = ConceptTransformationVisualizer(
            embedding_model="test-model",
            openai_api_key="test-key"
        )
        
        # Assertions
        mock_sentence_transformer.assert_called_once_with("test-model")
        mock_openai.assert_called_once()
        self.assertEqual(visualizer.llm_client, mock_openai_instance)

if __name__ == '__main__':
    unittest.main()