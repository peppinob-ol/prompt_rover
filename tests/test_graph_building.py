# /tests/test_graph_building.py
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np
import networkx as nx

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import ConceptTransformationVisualizer

class TestGraphBuilding(unittest.TestCase):
    
    @patch('main.SentenceTransformer')
    def setUp(self, mock_sentence_transformer):
        """Set up the test environment."""
        # Create a mock for the encoder
        mock_encoder = MagicMock()
        mock_sentence_transformer.return_value = mock_encoder
        
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
        
    def test_build_concept_graph_knn(self):
        """Test building a k-NN graph."""
        # Build k-NN graph
        graph = self.visualizer.build_concept_graph(
            self.test_concepts,
            k=2,
            use_mst=False,
            show_io_links=False
        )
        
        # Check graph properties
        self.assertEqual(type(graph), nx.Graph)
        self.assertEqual(len(graph.nodes), 3)
        
        # Each node should have at most 2 neighbors (k=2)
        for node in graph.nodes:
            self.assertLessEqual(len(list(graph.neighbors(node))), 2)
            
        # Check node attributes
        for concept in self.test_concepts:
            node = concept["label"]
            self.assertIn(node, graph.nodes)
            self.assertEqual(graph.nodes[node]["category"], concept["category"])
            self.assertEqual(graph.nodes[node]["description"], concept["description"])
            self.assertEqual(graph.nodes[node]["content_type"], concept["content_type"])
            np.testing.assert_array_equal(graph.nodes[node]["embedding"], concept["embedding"])
            
    def test_build_concept_graph_mst(self):
        """Test building a Minimum Spanning Tree graph."""
        # Build MST graph
        graph = self.visualizer.build_concept_graph(
            self.test_concepts,
            use_mst=True,
            show_io_links=False
        )
        
        # Check graph properties
        self.assertEqual(type(graph), nx.Graph)
        self.assertEqual(len(graph.nodes), 3)
        
        # MST should have exactly n-1 edges for n nodes
        self.assertEqual(len(graph.edges), 2)
        
        # Check that the graph is connected
        self.assertTrue(nx.is_connected(graph))
        
    def test_build_concept_graph_io_links(self):
        """Test building a graph with input-output links."""
        # Build graph with I/O links
        graph = self.visualizer.build_concept_graph(
            self.test_concepts,
            use_mst=True,
            show_io_links=True
        )
        
        # Check if there are edges between input and output concepts
        input_nodes = ["concept1", "concept2"]
        output_nodes = ["concept3"]
        
        # There should be at least one edge between input and output
        io_edges = [(u, v) for u, v in graph.edges if 
                   (u in input_nodes and v in output_nodes) or 
                   (u in output_nodes and v in input_nodes)]
        
        self.assertGreaterEqual(len(io_edges), 1)
        
    def test_build_concept_graph_with_messages(self):
        """Test building a graph with message linking."""
        # Create concepts with message IDs
        concepts_with_messages = [
            {
                "label": "concept1",
                "category": "entity",
                "description": "description 1",
                "content_type": "user",
                "message_id": 1,
                "embedding": np.array([0.1, 0.2, 0.3])
            },
            {
                "label": "concept2",
                "category": "process",
                "description": "description 2",
                "content_type": "user",
                "message_id": 1,
                "embedding": np.array([0.4, 0.5, 0.6])
            },
            {
                "label": "concept3",
                "category": "attribute",
                "description": "description 3",
                "content_type": "assistant",
                "message_id": 2,
                "embedding": np.array([0.7, 0.8, 0.9])
            }
        ]
        
        # Build graph with message linking
        graph = self.visualizer.build_concept_graph(
            concepts_with_messages,
            use_mst=True,
            show_io_links=False
        )
        
        # Check that concepts from the same message are connected
        self.assertTrue(graph.has_edge("concept1", "concept2"))
        
        # Check edge attributes for same-message connection
        if graph.has_edge("concept1", "concept2"):
            self.assertEqual(graph.edges["concept1", "concept2"].get("type"), "same_message")
        
    def test_build_concept_graph_single_concept(self):
        """Test building a graph with only one concept."""
        # Create a single concept
        single_concept = [self.test_concepts[0]]
        
        # Build graph
        graph = self.visualizer.build_concept_graph(single_concept)
        
        # Check graph properties
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(len(graph.edges), 0)
        self.assertIn("concept1", graph.nodes)

if __name__ == '__main__':
    unittest.main()