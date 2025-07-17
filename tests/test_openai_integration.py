# /tests/test_openai_integration.py
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Aggiungi il percorso del progetto al sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importa la classe da testare
from main import ConceptTransformationVisualizer

class TestOpenAIKeyIntegration(unittest.TestCase):
    """Test per verificare la corretta integrazione con OpenAI API"""
    
    def setUp(self):
        """Inizializza l'ambiente di test"""
        self.test_api_key = "sk-test12345"
        self.visualizer = ConceptTransformationVisualizer()
    
    @patch('openai.OpenAI')
    def test_initialize_openai_client(self, mock_openai):
        """Verifica la corretta inizializzazione del client OpenAI"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Inizializza con API key di test
        result = self.visualizer.initialize_openai_client(self.test_api_key)
        
        # Verifica che l'inizializzazione sia avvenuta correttamente
        self.assertTrue(result)
        self.assertIsNotNone(self.visualizer.llm_client)
        mock_openai.assert_called_once()
    
    @patch('openai.OpenAI')
    def test_process_text_pair_respects_use_llm_flag(self, mock_openai):
        """Verifica che process_text_pair usi LLM quando richiesto"""
        # Configura il mock per OpenAI
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Configura la risposta mock per l'API OpenAI
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '[{"label": "Test", "category": "test", "description": "Test"}]'
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Inizializza il client
        self.visualizer.initialize_openai_client(self.test_api_key)
        
        # Patch il metodo extract_concepts_with_llm per verificare che venga chiamato
        with patch.object(self.visualizer, 'extract_concepts_with_llm') as mock_extract_llm:
            # Configura il mock per evitare errori nelle chiamate successive
            mock_extract_llm.return_value = [
                {"label": "Test", "category": "test", "description": "Test description", "content_type": "user"}
            ]
            
            # Patch anche gli altri metodi necessari
            with patch.object(self.visualizer, 'compute_embeddings', return_value=[]):
                with patch.object(self.visualizer, 'build_concept_graph'):
                    with patch.object(self.visualizer, 'reduce_dimensions'):
                        # Esegui process_text_pair con use_llm=True
                        self.visualizer.process_text_pair(
                            "Testo input di test", 
                            "Testo output di test",
                            use_llm=True
                        )
            
            # Verifica che extract_concepts_with_llm sia stato chiamato
            mock_extract_llm.assert_called()
    
    def test_extract_concepts_with_llm_parameter_mismatch(self):
        """Verifica che non ci siano errori nei parametri di extract_concepts_with_llm"""
        with patch('openai.OpenAI'):
            self.visualizer.initialize_openai_client(self.test_api_key)
            
            # Patch la chiamata API per evitare chiamate reali
            with patch.object(self.visualizer.llm_client, 'chat') as mock_chat:
                mock_completions = MagicMock()
                mock_chat.completions = mock_completions
                
                mock_response = MagicMock()
                mock_choice = MagicMock()
                mock_choice.message.content = '[{"label": "Test", "category": "test", "description": "Test"}]'
                mock_response.choices = [mock_choice]
                mock_completions.create.return_value = mock_response
                
                # Verifica che il metodo non generi errori con i parametri attuali
                try:
                    concepts = self.visualizer.extract_concepts_with_llm("Test text", True)
                    self.assertIsInstance(concepts, list)
                except NameError as e:
                    if "source_label" in str(e):
                        self.fail("Bug identificato: 'source_label' non è definito ma viene utilizzato")
                    else:
                        self.fail(f"NameError inatteso: {e}")
                except Exception as e:
                    self.fail(f"Eccezione inattesa: {e}")

if __name__ == '__main__':
    unittest.main()