"""
Test di integrazione per Prompt Rover
"""

import pytest
from unittest.mock import patch
import numpy as np
from prompt_rover import ConceptTransformationVisualizer

class TestIntegration:
    """Test di integrazione end-to-end"""
    
    @patch('prompt_rover.core.embeddings.SentenceTransformer')
    def test_text_pair_processing(self, mock_transformer):
        """Test elaborazione completa di una coppia di testi"""
        # Configura mock
        mock_model = mock_transformer.return_value
        # Restituisci un numero dinamico di embeddings basato sull'input
        def mock_encode(texts):
            return np.random.rand(len(texts), 384)
        mock_model.encode.side_effect = mock_encode
        
        # Crea visualizzatore
        visualizer = ConceptTransformationVisualizer()
        
        # Testi di test
        input_text = "L'intelligenza artificiale è il futuro"
        output_text = "AI e machine learning trasformano il mondo"
        
        # Elabora
        df, fig, status = visualizer.process_text_pair(
            input_text, output_text,
            use_llm=False,
            dim_reduction="pca"
        )
        
        # Verifica risultati
        assert df is not None
        assert len(df) > 0
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert 'label' in df.columns
        assert fig is not None
        assert "successo" in status.lower()
    
    @patch('prompt_rover.core.embeddings.SentenceTransformer')
    def test_chat_mode_integration(self, mock_transformer):
        """Test modalità chat completa"""
        # Configura mock
        mock_model = mock_transformer.return_value
        mock_model.encode.return_value = np.random.rand(3, 384)
        
        # Crea visualizzatore
        visualizer = ConceptTransformationVisualizer()
        visualizer.initialize_chat_mode()
        
        # Primo messaggio
        df1, fig1, status1 = visualizer.process_new_message(
            "Parliamo di AI",
            is_user=True,
            use_llm=False
        )
        
        assert df1 is not None
        assert len(df1) > 0
        assert visualizer.message_counter == 1
        
        # Simula risposta
        mock_model.encode.return_value = np.random.rand(3, 384)
        df2, fig2, status2 = visualizer.process_new_message(
            "L'AI è affascinante",
            is_user=False,
            use_llm=False
        )
        
        assert df2 is not None
        assert len(df2) > len(df1)  # Più concetti
        assert visualizer.message_counter == 2
    
    def test_cache_integration(self):
        """Test che la cache funzioni nell'integrazione"""
        with patch('prompt_rover.core.embeddings.SentenceTransformer') as mock_transformer:
            mock_model = mock_transformer.return_value
            mock_model.encode.return_value = np.random.rand(3, 384)
            
            visualizer = ConceptTransformationVisualizer()
            
            # Prima elaborazione
            text = "Test cache"
            visualizer.process_text_pair(text, "output", use_llm=False)
            first_call_count = mock_model.encode.call_count
            
            # Seconda elaborazione con stesso testo
            visualizer.process_text_pair(text, "output", use_llm=False)
            second_call_count = mock_model.encode.call_count
            
            # La cache dovrebbe ridurre le chiamate
            assert second_call_count < first_call_count * 2
    
    @patch('openai.OpenAI')
    @patch('prompt_rover.core.embeddings.SentenceTransformer')
    def test_with_llm_integration(self, mock_transformer, mock_openai):
        """Test integrazione con LLM"""
        # Configura mocks
        mock_model = mock_transformer.return_value
        mock_model.encode.return_value = np.random.rand(2, 384)
        
        mock_client = mock_openai.return_value
        mock_response = mock_client.chat.completions.create.return_value
        mock_response.choices = [type('obj', (object,), {
            'message': type('obj', (object,), {
                'content': '[{"label": "AI", "category": "tech", "description": "Artificial Intelligence"}]'
            })()
        })()]
        
        # Test
        visualizer = ConceptTransformationVisualizer(openai_api_key="test-key")
        df, fig, status = visualizer.process_text_pair(
            "AI test", "ML test",
            use_llm=True
        )
        
        assert df is not None
        assert len(df) > 0
        assert mock_client.chat.completions.create.called
    
    def test_error_handling_integration(self):
        """Test gestione errori nell'integrazione"""
        visualizer = ConceptTransformationVisualizer()
        
        # Test con testi vuoti
        df, fig, status = visualizer.process_text_pair("", "", use_llm=False)
        assert "errore" in status.lower() or len(df) == 0
        
        # Test con parametri invalidi
        df, fig, status = visualizer.process_text_pair(
            "test", "test",
            use_llm=False,
            dim_reduction="invalid_method"
        )
        # Non dovrebbe crashare
        assert df is not None or "errore" in status.lower()