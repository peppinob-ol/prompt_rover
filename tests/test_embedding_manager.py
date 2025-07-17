"""
Test per il modulo EmbeddingManager
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from prompt_rover.core import EmbeddingManager

class TestEmbeddingManager:
    """Test per EmbeddingManager"""
    
    @patch('prompt_rover.core.embeddings.SentenceTransformer')
    def test_initialization(self, mock_transformer):
        """Test inizializzazione"""
        manager = EmbeddingManager()
        mock_transformer.assert_called_once()
        assert manager.model is not None
        assert hasattr(manager, 'cache')
        assert hasattr(manager, 'name_embeddings')
        assert hasattr(manager, 'description_embeddings')
    
    @patch('prompt_rover.core.embeddings.SentenceTransformer')
    def test_compute_embeddings_empty_list(self, mock_transformer):
        """Test con lista vuota di concetti"""
        manager = EmbeddingManager()
        result = manager.compute_embeddings([])
        assert result == []
    
    @patch('prompt_rover.core.embeddings.SentenceTransformer')
    def test_compute_embeddings_single_concept(self, mock_transformer):
        """Test con un singolo concetto"""
        # Configura mock
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        # Mock embeddings
        mock_embedding = np.random.rand(384)
        mock_model.encode.return_value = np.array([mock_embedding])
        
        manager = EmbeddingManager()
        concepts = [{
            'label': 'AI',
            'description': 'Artificial Intelligence',
            'category': 'tech'
        }]
        
        result = manager.compute_embeddings(concepts)
        
        assert len(result) == 1
        assert 'embedding' in result[0]
        assert isinstance(result[0]['embedding'], np.ndarray)
    
    @patch('prompt_rover.core.embeddings.SentenceTransformer')
    def test_compute_embeddings_with_weight(self, mock_transformer):
        """Test calcolo embeddings con peso"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        # Mock embeddings diversi per nome e descrizione
        name_emb = np.array([1.0, 0.0, 0.0])
        desc_emb = np.array([0.0, 1.0, 0.0])
        mock_model.encode.side_effect = [
            np.array([name_emb]),  # Per i nomi
            np.array([desc_emb])   # Per le descrizioni
        ]
        
        manager = EmbeddingManager()
        concepts = [{'label': 'test', 'description': 'test desc'}]
        
        # Test con peso 1.0 (solo nome)
        result = manager.compute_embeddings(concepts, name_weight=1.0)
        expected = name_emb / np.linalg.norm(name_emb)
        np.testing.assert_array_almost_equal(result[0]['embedding'], expected)
        
        # Reset mock
        mock_model.encode.side_effect = [
            np.array([name_emb]),
            np.array([desc_emb])
        ]
        
        # Test con peso 0.0 (solo descrizione)
        result = manager.compute_embeddings(concepts, name_weight=0.0)
        expected = desc_emb / np.linalg.norm(desc_emb)
        np.testing.assert_array_almost_equal(result[0]['embedding'], expected)
    
    @patch('prompt_rover.core.embeddings.SentenceTransformer')
    def test_embedding_caching(self, mock_transformer):
        """Test che gli embeddings vengano cachati"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        manager = EmbeddingManager()
        text = "test text"
        
        # Prima chiamata
        manager._encode_batch([text], prefix="test")
        assert mock_model.encode.call_count == 1
        
        # Seconda chiamata - dovrebbe usare cache
        manager._encode_batch([text], prefix="test")
        assert mock_model.encode.call_count == 1  # Non dovrebbe aumentare
    
    def test_get_similarity(self):
        """Test calcolo similarità"""
        manager = EmbeddingManager.__new__(EmbeddingManager)  # Skip init
        
        # Vettori ortogonali
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        sim = manager.get_similarity(emb1, emb2)
        assert sim == pytest.approx(0.0)
        
        # Vettori identici
        emb3 = np.array([1.0, 1.0, 1.0])
        sim = manager.get_similarity(emb3, emb3)
        assert sim == pytest.approx(1.0)
        
        # Vettori opposti
        emb4 = np.array([1.0, 0.0, 0.0])
        emb5 = np.array([-1.0, 0.0, 0.0])
        sim = manager.get_similarity(emb4, emb5)
        assert sim == pytest.approx(-1.0)
    
    def test_find_similar_concepts(self):
        """Test ricerca concetti simili"""
        manager = EmbeddingManager.__new__(EmbeddingManager)
        
        # Prepara embeddings di test
        concept_embeddings = {
            'A': np.array([1.0, 0.0, 0.0]),
            'B': np.array([0.9, 0.1, 0.0]),  # Simile ad A
            'C': np.array([0.0, 1.0, 0.0]),  # Diverso
            'D': np.array([0.8, 0.2, 0.0])   # Simile ad A
        }
        
        target = np.array([1.0, 0.0, 0.0])
        similar = manager.find_similar_concepts(target, concept_embeddings, top_k=2)
        
        assert len(similar) == 2
        assert similar[0][0] == 'A'  # Più simile
        assert similar[1][0] == 'B'  # Secondo più simile 