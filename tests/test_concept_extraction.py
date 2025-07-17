# /tests/test_concept_extraction.py
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import json

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import ConceptTransformationVisualizer

class TestConceptExtraction(unittest.TestCase):
    
    @patch('main.SentenceTransformer')
    def setUp(self, mock_sentence_transformer):
        """Set up the test."""
        # Setup the mock
        mock_encoder = MagicMock()
        mock_sentence_transformer.return_value = mock_encoder
        
        # Initialize the visualizer
        self.visualizer = ConceptTransformationVisualizer(embedding_model="test-model")
        
    @patch('main.spacy.load')
    def test_extract_concepts_alternative(self, mock_spacy_load):
        """Test extract_concepts_alternative method."""
        # Setup the mock
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        
        # Create mock entities
        mock_entity1 = MagicMock()
        mock_entity1.text = "entity1"
        mock_entity1.label_ = "PERSON"
        
        mock_entity2 = MagicMock()
        mock_entity2.text = "entity2"
        mock_entity2.label_ = "ORG"
        
        mock_doc.ents = [mock_entity1, mock_entity2]
        
        # Create mock noun chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "noun chunk 1"
        
        mock_chunk2 = MagicMock()
        mock_chunk2.text = "noun chunk 2"
        
        mock_doc.noun_chunks = [mock_chunk1, mock_chunk2]
        
        # Setup token mocks
        mock_token1 = MagicMock()
        mock_token1.pos_ = "NOUN"
        mock_token1.text = "token1"
        
        mock_token2 = MagicMock()
        mock_token2.pos_ = "PROPN"
        mock_token2.text = "token2"
        
        mock_doc.__iter__.return_value = [mock_token1, mock_token2]
        
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        # Extract concepts
        concepts = self.visualizer.extract_concepts_alternative("test text", True)
        
        # Assertions
        mock_spacy_load.assert_called_once()
        mock_nlp.assert_called_once_with("test text")
        
        # Should have extracted entities and noun chunks
        self.assertGreaterEqual(len(concepts), 2)  # At least 2 concepts
        
        # Check entity extraction
        entity_labels = [concept["label"] for concept in concepts]
        self.assertIn("entity1", entity_labels)
        self.assertIn("entity2", entity_labels)
        
        # Check noun chunk extraction
        self.assertIn("noun chunk 1", entity_labels)
        self.assertIn("noun chunk 2", entity_labels)