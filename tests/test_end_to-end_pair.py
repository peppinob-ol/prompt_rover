# /tests/test_end_to_end_pair.py
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np
import pandas as pd

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import ConceptTransformationVisualizer

class TestEndToEndPair(unittest.TestCase):
    
    @patch('main.SentenceTransformer')
    @patch('main.UMAP')
    @patch('openai.OpenAI')
    def setUp(self, mock_openai, mock_umap, mock_sentence_transformer):
        """Set up the test environment."""
        # Create a mock for the encoder
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_sentence_transformer.return_value = mock_encoder
        
        # Set up mock for UMAP
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_umap.return_value = mock_umap_instance
        
        # Set up mock for OpenAI
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        # Set up mock for chat completions
        mock_chat = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = mock_chat
        
        # Set up mock for chat response
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = """```json
        [
            {
                "label": "concept1",
                "category": "entity",
                "description": "description 1"
            }
        ]
        ```"""
        mock_choice.message = mock_message
        mock_chat.choices = [mock_choice]
        
        # Initialize the visualizer
        self.visualizer = ConceptTransformationVisualizer(
            embedding_model="test-model",
            openai_api_key="test-key"
        )
        
    @patch.object(ConceptTransformationVisualizer, 'visualize_concepts_interactive')
    def test_process_text_pair_with_llm(self, mock_visualize):
        """Test the complete process_text_pair flow with LLM."""
        # Set up mock for visualization
        mock_visualize.return_value = "mock_figure"
        
        # Process a text pair
        df, fig, status = self.visualizer.process_text_pair(
            "input text",
            "output text",
            use_llm=True,
            name_weight=0.8,
            dim_reduction="umap",
            use_mst=True,
            show_io_links=True
        )
        
        # Check results
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(fig, "mock_figure")
        self.assertIn("Processing completed", status)
        
        # Check that OpenAI was called
        self.visualizer.llm_client.chat.completions.create.assert_called()
        
    @patch.object(ConceptTransformationVisualizer, 'extract_concepts_alternative')
    @patch.object(ConceptTransformationVisualizer, 'visualize_concepts_interactive')
    def test_process_text_pair_without_llm(self, mock_visualize, mock_extract):
        """Test the complete process_text_pair flow without LLM."""
        # Set up mocks
        mock_extract.return_value = [{"label": "concept1"}]
        mock_visualize.return_value = "mock_figure"
        
        # Process a text pair
        df, fig, status = self.visualizer.process_text_pair(
            "input text",
            "output text",
            use_llm=False,
            name_weight=0.8,
            dim_reduction="tsne",
            use_mst=False,
            show_io_links=False
        )
        
        # Check results
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(fig, "mock_figure")
        self.assertIn("Processing completed", status)
        
        # Check that alternative extraction was used
        mock_extract.assert_called()
        
    @patch.object(ConceptTransformationVisualizer, 'extract_concepts_with_llm')
    @patch.object(ConceptTransformationVisualizer, 'extract_concepts_alternative')
    def test_process_text_pair_error_handling(self, mock_extract_alt, mock_extract_llm):
        """Test error handling in process_text_pair."""
        # Set up mock to raise an exception
        mock_extract_llm.side_effect = Exception("Test error")
        mock_extract_alt.side_effect = Exception("Another error")
        
        # Process a text pair with error
        df, fig, status = self.visualizer.process_text_pair(
            "input text",
            "output text",
            use_llm=True
        )
        
        # Check results reflect error
        self.assertEqual(len(df), 0)
        self.assertIsNotNone(fig)
        self.assertIn("Error", status)

if __name__ == '__main__':
    unittest.main()