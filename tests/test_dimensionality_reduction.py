# /tests/test_dimensionality_reduction.py
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np
import pandas as pd

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import ConceptTransformationVisualizer

class TestDimensionalityReduction(unittest.TestCase):
    
    @patch('main.SentenceTransformer')
    @patch('main.UMAP')
    @patch('main.TSNE')
    @patch('main.PCA')
    def setUp(self, mock_pca, mock_tsne, mock_umap, mock_sentence_transformer):
        """Set up the test environment."""
        # Create a mock for the encoder
        mock_encoder = MagicMock()
        mock_sentence_transformer.return_value = mock_encoder
        
        # Set up mock for dimensional reducers
        self.mock_umap_instance = MagicMock()
        self.mock_umap_instance.fit_transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mock_umap.return_value = self.mock_umap_instance
        
        self.mock_tsne_instance = MagicMock()
        self.mock_tsne_instance.fit_transform.return_value = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
        # Configure t-SNE mock to accept any perplexity value
        self.mock_tsne_instance.fit_transform.side_effect = lambda X, **kwargs: np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
        mock_tsne.return_value = self.mock_tsne_instance
        
        self.mock_pca_instance = MagicMock()
        self.mock_pca_instance.transform.return_value = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        mock_pca.return_value = self.mock_pca_instance
        
        # Initialize the visualizer
        self.visualizer = ConceptTransformationVisualizer(embedding_model="test-model")
        
        # Test concepts with embeddings
        self.test_concepts = [
            {
                "label": "concept1",
                "category": "entity",
                "description": "description 1",
                "content_type": "user",
                "embedding": np.array([0.1, 0.2, 0.3])
            },
            {
                "label": "concept2",
                "category": "process",
                "description": "description 2",
                "content_type": "user",
                "embedding": np.array([0.4, 0.5, 0.6])
            },
            {
                "label": "concept3",
                "category": "attribute",
                "description": "description 3",
                "content_type": "assistant",
                "embedding": np.array([0.7, 0.8, 0.9])
            }
        ]
        
    def test_reduce_dimensions_umap(self):
        """Test dimensionality reduction with UMAP."""
        # Reduce dimensions with UMAP
        df = self.visualizer.reduce_dimensions(
            self.test_concepts,
            method="umap",
            n_neighbors=2,
            min_dist=0.1
        )
        
        # Check that UMAP was called with correct parameters
        self.mock_umap_instance.fit_transform.assert_called_once()
        call_args = self.mock_umap_instance.fit_transform.call_args[1]
        self.assertEqual(call_args.get('n_neighbors'), 2)
        self.assertEqual(call_args.get('min_dist'), 0.1)
        
        # Check dataframe structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn("label", df.columns)
        self.assertIn("category", df.columns)
        self.assertIn("description", df.columns)
        self.assertIn("source", df.columns)
        self.assertIn("x", df.columns)
        self.assertIn("y", df.columns)
        
        # Check coordinates
        self.assertEqual(df.iloc[0]["x"], 1.0)
        self.assertEqual(df.iloc[0]["y"], 2.0)
        
    def test_reduce_dimensions_tsne(self):
        """Test dimensionality reduction with t-SNE."""
        # Reduce dimensions with t-SNE
        df = self.visualizer.reduce_dimensions(
            self.test_concepts,
            method="tsne",
            perplexity=1  # Use a valid perplexity value for 3 samples
        )
        
        # Check that t-SNE was called with correct parameters
        self.mock_tsne_instance.fit_transform.assert_called_once()
        call_args = self.mock_tsne_instance.fit_transform.call_args[1]
        self.assertEqual(call_args.get('perplexity'), 1)
        
        # Check dataframe structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn("label", df.columns)
        self.assertIn("category", df.columns)
        self.assertIn("description", df.columns)
        self.assertIn("source", df.columns)
        self.assertIn("x", df.columns)
        self.assertIn("y", df.columns)
        
        # Check coordinates
        self.assertEqual(df.iloc[0]["x"], 1.5)
        self.assertEqual(df.iloc[0]["y"], 2.5)
        
    def test_reduce_dimensions_pca(self):
        """Test dimensionality reduction with PCA."""
        # Reduce dimensions
        df = self.visualizer.reduce_dimensions(self.test_concepts, method="pca")
        
        # Check dataframe structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn("label", df.columns)
        self.assertIn("category", df.columns)
        self.assertIn("description", df.columns)
        self.assertIn("source", df.columns)
        self.assertIn("x", df.columns)
        self.assertIn("y", df.columns)
        
        # Check that coordinates are numeric
        self.assertTrue(isinstance(df.iloc[0]["x"], (int, float)))
        self.assertTrue(isinstance(df.iloc[0]["y"], (int, float)))
        
    def test_reduce_dimensions_empty(self):
        """Test dimensionality reduction with empty concept list."""
        # Reduce dimensions for an empty list
        df = self.visualizer.reduce_dimensions([])
        
        # Check that the result is an empty dataframe
        self.assertEqual(len(df), 0)
        
    def test_reduce_dimensions_single_concept(self):
        """Test dimensionality reduction with a single concept."""
        # Reduce dimensions for a single concept
        single_concept = [self.test_concepts[0]]
        df = self.visualizer.reduce_dimensions(single_concept)
        
        # Check that the result is a dataframe with one row
        self.assertEqual(len(df), 1)
        
        # For a single concept, coordinates should be [0, 0]
        self.assertEqual(df.iloc[0]["x"], 0.0)
        self.assertEqual(df.iloc[0]["y"], 0.0)
        
    def test_apply_umap(self):
        """Test the legacy apply_umap function."""
        # Call the legacy function
        df = self.visualizer.apply_umap(self.test_concepts)
        
        # Check that it returns the same result as reduce_dimensions
        df2 = self.visualizer.reduce_dimensions(self.test_concepts, method="umap")
        pd.testing.assert_frame_equal(df, df2)

if __name__ == '__main__':
    unittest.main()