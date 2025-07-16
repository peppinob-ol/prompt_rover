# /tests/test_mock_openai.py
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np
import json

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import ConceptTransformationVisualizer

class TestMockOpenAI(unittest.TestCase):
    
    @patch('main.SentenceTransformer')
    @patch('openai.OpenAI')
    def setUp(self, mock_openai, mock_sentence_transformer):
        """Set up the test."""
        # Setup the SentenceTransformer mock
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_sentence_transformer.return_value = mock_encoder
        
        # Setup the OpenAI mock
        self.mock_openai_instance = MagicMock()
        mock_openai.return_value = self.mock_openai_instance
        
        # Initialize the visualizer
        self.visualizer = ConceptTransformationVisualizer(
            embedding_model="test-model",
            openai_api_key="test-key"
        )
        
    def test_extract_concepts_with_llm_success(self):
        """Test successful concept extraction with LLM."""
        # Setup mock response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = """```json
        [
            {
                "label": "test concept",
                "category": "entity",
                "description": "test description"
            }
        ]
        ```"""
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        self.visualizer.llm_client.chat.completions.create.return_value = mock_response
        
        # Extract concepts
        concepts = self.visualizer.extract_concepts_with_llm("test text", True)
        
        # Assertions
        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0]["label"], "test concept")
        self.assertEqual(concepts[0]["category"], "entity")
        self.assertEqual(concepts[0]["description"], "test description")
        self.assertEqual(concepts[0]["content_type"], "user")
        
    def test_extract_concepts_with_llm_json_error(self):
        """Test handling of JSON parsing error in LLM response."""
        # Setup mock response with invalid JSON
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = """```
        Invalid JSON
        ```"""
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        self.visualizer.llm_client.chat.completions.create.return_value = mock_response
        
        # Mock the alternative method to check if it's called
        with patch.object(self.visualizer, 'extract_concepts_alternative') as mock_alternative:
            mock_alternative.return_value = [{"label": "alternative concept"}]
            
            # Extract concepts
            concepts = self.visualizer.extract_concepts_with_llm("test text", True)
            
            # Assertions
            mock_alternative.assert_called_once()
            self.assertEqual(len(concepts), 1)
            self.assertEqual(concepts[0]["label"], "alternative concept")