# /tests/test_end_to_end_chat.py
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np
import pandas as pd

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import ConceptTransformationVisualizer

class TestEndToEndChat(unittest.TestCase):
    
    @patch('main.SentenceTransformer')
    @patch('openai.OpenAI')
    def setUp(self, mock_openai, mock_sentence_transformer):
        """Set up the test environment."""
        # Create a mock for the encoder
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_sentence_transformer.return_value = mock_encoder
        
        # Set up mock for OpenAI
        self.mock_openai_instance = MagicMock()
        mock_openai.return_value = self.mock_openai_instance
        
        # Initialize the visualizer
        self.visualizer = ConceptTransformationVisualizer(
            embedding_model="test-model",
            openai_api_key="test-key"
        )
        
        # Initialize chat mode
        self.visualizer.initialize_chat_mode()
        
    @patch.object(ConceptTransformationVisualizer, 'extract_concepts_with_llm')
    @patch.object(ConceptTransformationVisualizer, 'visualize_concepts_interactive')
    def test_process_new_message(self, mock_visualize, mock_extract):
        """Test processing a new message in chat mode."""
        # Set up mocks
        mock_extract.return_value = [{"label": "concept1", "category": "entity", "description": "description"}]
        mock_visualize.return_value = "mock_figure"
        
        # Process a user message
        df, fig, status = self.visualizer.process_new_message(
            "hello world",
            is_user=True,
            use_llm=True
        )
        
        # Check results
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(fig, "mock_figure")
        self.assertIn("Message processed", status)
        
        # Check that chat history was updated
        self.assertEqual(len(self.visualizer.chat_history), 1)
        self.assertEqual(self.visualizer.chat_history[0]["content"], "hello world")
        
        # Check that concepts were added
        self.assertEqual(len(self.visualizer.chat_concepts), 1)
        
    def test_generate_chat_response(self):
        """Test generating a chat response."""
        # Set up mock for OpenAI response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Hello, how can I help you?"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        self.mock_openai_instance.chat.completions.create.return_value = mock_response
        
        # Generate a response
        history = [{"role": "user", "content": "Hello"}]
        response = self.visualizer.generate_chat_response(history)
        
        # Check response
        self.assertEqual(response, "Hello, how can I help you?")
        
        # Check that OpenAI was called
        self.mock_openai_instance.chat.completions.create.assert_called_once()
        
    def test_initialize_chat_mode(self):
        """Test initializing chat mode."""
        # Add some data to the visualizer
        self.visualizer.chat_concepts = ["test"]
        self.visualizer.chat_history = ["test"]
        self.visualizer.name_embeddings = {"test": "test"}
        
        # Initialize chat mode
        result = self.visualizer.initialize_chat_mode()
        
        # Check that data was reset
        self.assertEqual(self.visualizer.chat_concepts, [])
        self.assertEqual(self.visualizer.chat_history, [])
        self.assertEqual(self.visualizer.name_embeddings, {})
        
        # Check result message
        self.assertEqual(result, "Chat mode initialized")
        
    @patch.object(ConceptTransformationVisualizer, 'extract_concepts_alternative')
    def test_chat_with_many_concepts_limit(self, mock_extract):
        """Test the limit on the number of concepts in chat mode."""
        # Set up mock to return many concepts
        mock_extract.return_value = [{"label": f"concept{i}", "category": "entity", "description": "description", "embedding": np.array([0.1, 0.2, 0.3])} for i in range(50)]
        
        # Add many concepts to reach the limit
        self.visualizer.chat_concepts = [{"label": f"existing{i}", "category": "entity", "description": "description", "embedding": np.array([0.1, 0.2, 0.3])} for i in range(450)]
        
        # Process a message that would exceed the limit
        df, fig, status = self.visualizer.process_new_message(
            "too many concepts",
            is_user=True,
            use_llm=False
        )
        
        # Check that the limit message is returned
        self.assertIn("Concept limit reached", status)
        
        # Check that no new concepts were added
        self.assertEqual(len(self.visualizer.chat_concepts), 450)
        
        # Check that the mock was called
        mock_extract.assert_called_once()

if __name__ == '__main__':
    unittest.main()