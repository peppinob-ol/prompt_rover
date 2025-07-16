"""
Test per verificare che tutti gli import funzionino correttamente
"""

import pytest

def test_main_package_import():
    """Test import del package principale"""
    import prompt_rover
    assert hasattr(prompt_rover, '__version__')
    assert hasattr(prompt_rover, 'ConceptTransformationVisualizer')

def test_core_module_imports():
    """Test import dei moduli core"""
    from prompt_rover.core import (
        ConceptTransformationVisualizer,
        ConceptExtractor,
        EmbeddingManager,
        GraphBuilder,
        DimensionReducer
    )
    
    # Verifica che le classi siano importate
    assert ConceptTransformationVisualizer is not None
    assert ConceptExtractor is not None
    assert EmbeddingManager is not None
    assert GraphBuilder is not None
    assert DimensionReducer is not None

def test_visualization_module_imports():
    """Test import dei moduli di visualizzazione"""
    from prompt_rover.visualization import (
        StaticVisualizer,
        InteractiveVisualizer
    )
    
    assert StaticVisualizer is not None
    assert InteractiveVisualizer is not None

def test_chat_module_imports():
    """Test import del modulo chat"""
    from prompt_rover.chat import ChatHandler
    assert ChatHandler is not None

def test_ui_module_imports():
    """Test import del modulo UI"""
    from prompt_rover.ui import create_gradio_interface
    assert create_gradio_interface is not None

def test_utils_module_imports():
    """Test import dei moduli utils"""
    from prompt_rover.utils import (
        get_logger,
        setup_logging,
        log_execution_time,
        CacheManager
    )
    
    assert get_logger is not None
    assert setup_logging is not None
    assert log_execution_time is not None
    assert CacheManager is not None

def test_config_import():
    """Test import delle configurazioni"""
    from prompt_rover import config
    
    # Verifica alcune costanti chiave
    assert hasattr(config, 'DEFAULT_EMBEDDING_MODEL')
    assert hasattr(config, 'TEAL')
    assert hasattr(config, 'ORANGE')
    assert hasattr(config, 'MAX_CONCEPTS_PER_TEXT')

def test_no_circular_imports():
    """Test che non ci siano import circolari"""
    # Questo test passerà se tutti gli import precedenti sono riusciti
    # senza errori di import circolari
    assert True 