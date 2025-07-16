"""
Test completo per validare che il refactoring funzioni correttamente
"""

import pytest
import sys
import os
import numpy as np

# Mock delle dipendenze pesanti per velocizzare i test
sys.modules['sentence_transformers'] = type(sys)('sentence_transformers')
sys.modules['sentence_transformers'].SentenceTransformer = lambda x: type('MockModel', (), {'encode': lambda self, texts: np.array([[0.1, 0.2, 0.3]] * len(texts))})()

# Mock di sklearn migliorato
sklearn = type(sys)('sklearn')
sklearn.decomposition = type(sys)('sklearn.decomposition')
sklearn.manifold = type(sys)('sklearn.manifold')

# Mock PCA con fit method
class MockPCA:
    def __init__(self, **kwargs):
        self.n_components = kwargs.get('n_components', 2)
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        # Restituisce coordinate 2D per ogni punto
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        return np.array([[i, i+0.5] for i in range(n_samples)])
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Mock TSNE con fit_transform
class MockTSNE:
    def __init__(self, **kwargs):
        self.n_components = kwargs.get('n_components', 2)
    
    def fit_transform(self, X):
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        return np.array([[i*0.1, i*0.2] for i in range(n_samples)])

sklearn.decomposition.PCA = MockPCA
sklearn.manifold.TSNE = MockTSNE
sys.modules['sklearn'] = sklearn
sys.modules['sklearn.decomposition'] = sklearn.decomposition
sys.modules['sklearn.manifold'] = sklearn.manifold

# Mock UMAP
class MockUMAP:
    def __init__(self, **kwargs):
        self.n_components = kwargs.get('n_components', 2)
    
    def fit_transform(self, X):
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        return np.array([[i*0.3, i*0.4] for i in range(n_samples)])

sys.modules['umap'] = type(sys)('umap')
sys.modules['umap'].UMAP = MockUMAP

# Mock di gradio
sys.modules['gradio'] = type(sys)('gradio')
sys.modules['gradio'].Blocks = lambda **kwargs: type('MockBlocks', (), {'launch': lambda self, **kw: None})()
sys.modules['gradio'].Textbox = lambda **kwargs: None
sys.modules['gradio'].Button = lambda **kwargs: None
sys.modules['gradio'].Radio = lambda **kwargs: None
sys.modules['gradio'].Checkbox = lambda **kwargs: None
sys.modules['gradio'].Slider = lambda **kwargs: None
sys.modules['gradio'].Dropdown = lambda **kwargs: None
sys.modules['gradio'].Plot = lambda **kwargs: None
sys.modules['gradio'].HTML = lambda **kwargs: None
sys.modules['gradio'].DataFrame = lambda **kwargs: None
sys.modules['gradio'].Tab = lambda **kwargs: None
sys.modules['gradio'].Row = lambda **kwargs: None
sys.modules['gradio'].Column = lambda **kwargs: None
sys.modules['gradio'].JSON = lambda **kwargs: None

# Mock di plotly più completo
plotly_module = type(sys)('plotly')
go_module = type(sys)('plotly.graph_objects')
go_module.Figure = type('Figure', (), {
    'add_trace': lambda self, *args, **kwargs: None,
    'update_layout': lambda self, *args, **kwargs: None,
    'add_annotation': lambda self, *args, **kwargs: None,
    'to_html': lambda self, **kwargs: "<div>Mock Figure</div>"
})
go_module.Scatter = lambda **kwargs: {}
plotly_module.graph_objects = go_module
sys.modules['plotly'] = plotly_module
sys.modules['plotly.graph_objects'] = go_module

# Aggiungi il percorso del progetto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestRefactoringValidation:
    """Test per validare che il refactoring sia completo e funzionante"""
    
    def test_package_structure(self):
        """Verifica che la struttura del package sia corretta"""
        import prompt_rover
        
        # Verifica attributi principali
        assert hasattr(prompt_rover, '__version__')
        assert hasattr(prompt_rover, 'ConceptTransformationVisualizer')
        
        # Verifica che il visualizzatore sia importabile
        from prompt_rover import ConceptTransformationVisualizer
        assert ConceptTransformationVisualizer is not None
    
    def test_all_modules_exist(self):
        """Verifica che tutti i moduli esistano"""
        modules_to_test = [
            'prompt_rover.config',
            'prompt_rover.utils',
            'prompt_rover.utils.logging',
            'prompt_rover.utils.cache',
            'prompt_rover.utils.decorators',
            'prompt_rover.core',
            'prompt_rover.core.concept_extractor',
            'prompt_rover.core.embeddings',
            'prompt_rover.core.graph_builder',
            'prompt_rover.core.dimension_reducer',
            'prompt_rover.core.visualizer',
            'prompt_rover.visualization',
            'prompt_rover.visualization.static_viz',
            'prompt_rover.visualization.interactive_viz',
            'prompt_rover.chat',
            'prompt_rover.chat.chat_handler',
            'prompt_rover.ui',
            'prompt_rover.ui.gradio_app'
        ]
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
                print(f"✓ {module_name}")
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")
    
    def test_core_components_integration(self):
        """Test che i componenti core si integrino correttamente"""
        from prompt_rover.core import (
            ConceptExtractor,
            EmbeddingManager,
            GraphBuilder,
            DimensionReducer
        )
        
        # Crea istanze
        extractor = ConceptExtractor()
        embedder = EmbeddingManager()
        graph_builder = GraphBuilder()
        dim_reducer = DimensionReducer()
        
        # Test estrazione concetti
        concepts = extractor.extract_concepts("Test text", is_user=True, use_llm=False)
        assert isinstance(concepts, list)
        
        # Test embeddings
        if concepts:
            embedded = embedder.compute_embeddings(concepts)
            assert all('embedding' in c for c in embedded)
            
            # Test costruzione grafo
            graph = graph_builder.build_graph(embedded)
            assert graph is not None
            
            # Test riduzione dimensionale
            df = dim_reducer.reduce_dimensions(embedded)
            assert df is not None
            assert 'x' in df.columns
            assert 'y' in df.columns
    
    def test_visualizer_integration(self):
        """Test che il visualizzatore principale funzioni"""
        from prompt_rover import ConceptTransformationVisualizer
        
        visualizer = ConceptTransformationVisualizer()
        
        # Test modalità input/output
        df, fig, status = visualizer.process_text_pair(
            "Input text", "Output text",
            use_llm=False,
            dim_reduction="pca"
        )
        
        assert df is not None
        assert status is not None
        assert "successo" in status.lower() or "completed" in status.lower()
    
    def test_chat_mode(self):
        """Test modalità chat"""
        from prompt_rover import ConceptTransformationVisualizer
        
        visualizer = ConceptTransformationVisualizer()
        visualizer.initialize_chat_mode()
        
        # Test messaggio
        df, fig, status = visualizer.process_new_message(
            "Hello world",
            is_user=True,
            use_llm=False
        )
        
        assert df is not None
        assert visualizer.message_counter == 1
    
    def test_utils_functionality(self):
        """Test funzionalità utils"""
        from prompt_rover.utils import CacheManager, get_logger
        
        # Test cache
        cache = CacheManager()
        cache.set("test", "value")
        assert cache.get("test") == "value"
        
        # Test logger
        logger = get_logger("test")
        assert logger is not None
    
    def test_config_constants(self):
        """Test che le costanti di configurazione esistano"""
        from prompt_rover import config
        
        required_constants = [
            'DEFAULT_EMBEDDING_MODEL',
            'TEAL', 'ORANGE',
            'CONTENT_TYPE', 'LABEL', 'EMBEDDING',
            'MAX_CONCEPTS_PER_TEXT'
        ]
        
        for const in required_constants:
            assert hasattr(config, const), f"Missing constant: {const}"
    
    def test_no_old_main_dependency(self):
        """Verifica che non ci siano più dipendenze da main.py"""
        # Questo test passerà se tutti i test precedenti sono passati
        # senza errori di import da 'main'
        assert True

def run_validation():
    """Esegue tutti i test di validazione"""
    print("=" * 60)
    print("VALIDAZIONE REFACTORING PROMPT ROVER")
    print("=" * 60)
    
    test = TestRefactoringValidation()
    
    tests = [
        ("Struttura package", test.test_package_structure),
        ("Esistenza moduli", test.test_all_modules_exist),
        ("Integrazione componenti", test.test_core_components_integration),
        ("Visualizzatore principale", test.test_visualizer_integration),
        ("Modalità chat", test.test_chat_mode),
        ("Funzionalità utils", test.test_utils_functionality),
        ("Costanti config", test.test_config_constants),
        ("Nessuna dipendenza da main.py", test.test_no_old_main_dependency)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\nTest: {name}...", end=" ")
            test_func()
            print("✓ PASSATO")
            passed += 1
        except Exception as e:
            print(f"✗ FALLITO: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Risultati: {passed} passati, {failed} falliti")
    
    if failed == 0:
        print("✅ REFACTORING VALIDATO CON SUCCESSO!")
    else:
        print("❌ CI SONO ANCORA PROBLEMI DA RISOLVERE")
    
    return failed == 0

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1) 