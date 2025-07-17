"""
Test per il modulo ConceptExtractor
"""

import pytest
from unittest.mock import Mock, patch
from prompt_rover.core import ConceptExtractor

class TestConceptExtractor:
    """Test per ConceptExtractor"""
    
    def test_initialization_without_api_key(self):
        """Test inizializzazione senza chiave API"""
        extractor = ConceptExtractor()
        assert extractor.llm_client is None
        assert extractor.nlp is None
    
    def test_initialization_with_api_key(self):
        """Test inizializzazione con chiave API"""
        with patch('openai.OpenAI') as mock_openai:
            extractor = ConceptExtractor("test-key")
            assert extractor.llm_client is not None
    
    def test_extract_concepts_alternative(self):
        """Test estrazione concetti senza LLM"""
        extractor = ConceptExtractor()
        text = "L'intelligenza artificiale sta trasformando il mondo della tecnologia."
        
        concepts = extractor.extract_concepts(text, is_user=True, use_llm=False)
        
        assert isinstance(concepts, list)
        assert len(concepts) > 0
        assert all('label' in c for c in concepts)
        assert all('category' in c for c in concepts)
        assert all('description' in c for c in concepts)
        assert all(c['content_type'] == 'user' for c in concepts)
    
    def test_extract_concepts_with_llm_fallback(self):
        """Test che l'estrazione con LLM cada su metodo alternativo senza client"""
        extractor = ConceptExtractor()
        text = "Machine learning e deep learning."
        
        # Senza client LLM, dovrebbe usare metodo alternativo
        concepts = extractor.extract_concepts(text, is_user=False, use_llm=True)
        
        assert isinstance(concepts, list)
        assert len(concepts) > 0
        assert all(c['content_type'] == 'assistant' for c in concepts)
    
    @patch('openai.OpenAI')
    def test_extract_concepts_with_llm_success(self, mock_openai_class):
        """Test estrazione concetti con LLM (simulato)"""
        # Configura mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''[
            {
                "label": "AI",
                "category": "technology",
                "description": "Artificial Intelligence"
            }
        ]'''
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test
        extractor = ConceptExtractor("test-key")
        concepts = extractor.extract_concepts("AI test", is_user=True, use_llm=True)
        
        assert len(concepts) == 1
        assert concepts[0]['label'] == 'AI'
        assert concepts[0]['content_type'] == 'user'
    
    def test_spacy_model_loading(self):
        """Test caricamento modello spaCy"""
        extractor = ConceptExtractor()
        nlp = extractor._load_spacy_model()
        
        assert nlp is not None
        assert extractor.nlp is not None
    
    def test_concept_deduplication(self):
        """Test deduplicazione concetti"""
        extractor = ConceptExtractor()
        text = "AI AI AI machine learning machine learning"
        
        concepts = extractor._extract_alternative(text, is_user=True)
        
        # Verifica che non ci siano duplicati
        labels = [c['label'].lower() for c in concepts]
        assert len(labels) == len(set(labels))
    
    def test_max_concepts_limit(self):
        """Test limite massimo concetti"""
        extractor = ConceptExtractor()
        # Testo lungo per generare molti concetti
        text = " ".join([f"concetto{i}" for i in range(50)])
        
        concepts = extractor._extract_alternative(text, is_user=True)
        
        assert len(concepts) <= 10  # MAX_CONCEPTS_PER_TEXT 