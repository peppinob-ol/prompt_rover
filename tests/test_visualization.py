# /tests/test_visualization.py
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import ConceptTransformationVisualizer

class TestVisualization(unittest.TestCase):
    
    @patch('main.SentenceTransformer')
    def setUp(self, mock_sentence_transformer):
        """Set up the test environment."""
        # Create a mock for the encoder
        mock_encoder = MagicMock()
        mock_sentence_transformer.return_value = mock_encoder
        
        # Initialize the visualizer
        self.visualizer = ConceptTransformationVisualizer(embedding_model="test-model")
        
        # Create a sample dataframe for visualization
        self.df = pd.DataFrame({
            "label": ["concept1", "concept2", "concept3"],
            "category": ["entity", "process", "attribute"],
            "description": ["description 1", "description 2", "description 3"],
            "source": ["user", "user", "assistant"],
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0]
        })
        
        # Create a sample graph
        G = nx.Graph()
        G.add_node("concept1", category="entity", description="description 1", content_type="user")
        G.add_node("concept2", category="process", description="description 2", content_type="user")
        G.add_node("concept3", category="attribute", description="description 3", content_type="assistant")
        G.add_edge("concept1", "concept2", weight=0.8)
        G.add_edge("concept2", "concept3", weight=0.5)
        self.visualizer.concept_graph = G
        
    def test_visualize_concepts(self):
        """Test the unified visualization function."""
        # Generate visualization
        fig = self.visualizer.visualize_concepts(self.df)
        
        # Check that a figure is returned
        self.assertIsInstance(fig, plt.Figure)
        
        # Close the figure to avoid warning
        plt.close(fig)
        
    def test_visualize_concepts_evolution(self):
        """Test visualization with evolution."""
        # Add message_id to dataframe
        df_with_messages = self.df.copy()
        df_with_messages["message_id"] = [1, 2, 3]
        
        # Generate visualization
        fig = self.visualizer.visualize_concepts(df_with_messages, show_evolution=True)
        
        # Check that a figure is returned
        self.assertIsInstance(fig, plt.Figure)
        
        # Close the figure to avoid warning
        plt.close(fig)
        
    def test_visualize_concepts_empty(self):
        """Test visualization with empty dataframe."""
        # Create empty dataframe
        empty_df = pd.DataFrame(columns=["label", "category", "description", "source", "x", "y"])
        
        # Generate visualization
        fig = self.visualizer.visualize_concepts(empty_df)
        
        # Check that a figure is returned
        self.assertIsInstance(fig, plt.Figure)
        
        # Close the figure to avoid warning
        plt.close(fig)
        
    def test_visualize_concepts_interactive(self):
        """Test the interactive visualization function."""
        # Generate interactive visualization
        fig = self.visualizer.visualize_concepts_interactive(self.df)
        
        # Check that a plotly figure is returned
        self.assertIsInstance(fig, go.Figure)
        self.assertIsNotNone(fig.data)  # Check that the figure has data
        
    def test_legacy_visualize_functions(self):
        """Test the legacy visualization functions."""
        # Test visualize_static
        fig1 = self.visualizer.visualize_static(self.df)
        self.assertIsInstance(fig1, plt.Figure)
        plt.close(fig1)
        
        # Test visualize_static_chat
        df_with_messages = self.df.copy()
        df_with_messages["message_id"] = [1, 2, 3]
        fig2 = self.visualizer.visualize_static_chat(df_with_messages)
        self.assertIsInstance(fig2, plt.Figure)
        plt.close(fig2)

if __name__ == '__main__':
    unittest.main()