"""
Test di base per verificare che il refactoring funzioni
"""

import sys
import os

# Aggiungi il percorso del progetto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Verifica che tutti i moduli si importino correttamente"""
    try:
        # Test import del package principale
        import prompt_rover
        print("✓ Package prompt_rover importato")
        
        # Test import dei moduli core
        from prompt_rover.core import ConceptTransformationVisualizer
        print("✓ ConceptTransformationVisualizer importato")
        
        from prompt_rover.core import ConceptExtractor
        print("✓ ConceptExtractor importato")
        
        from prompt_rover.core import EmbeddingManager
        print("✓ EmbeddingManager importato")
        
        from prompt_rover.core import GraphBuilder
        print("✓ GraphBuilder importato")
        
        from prompt_rover.core import DimensionReducer
        print("✓ DimensionReducer importato")
        
        # Test import moduli visualizzazione
        from prompt_rover.visualization import StaticVisualizer
        print("✓ StaticVisualizer importato")
        
        from prompt_rover.visualization import InteractiveVisualizer
        print("✓ InteractiveVisualizer importato")
        
        # Test import chat
        from prompt_rover.chat import ChatHandler
        print("✓ ChatHandler importato")
        
        # Test import UI
        from prompt_rover.ui import create_gradio_interface
        print("✓ create_gradio_interface importato")
        
        print("\n✅ Tutti gli import sono riusciti!")
        return True
        
    except Exception as e:
        print(f"\n❌ Errore negli import: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test funzionalità di base"""
    try:
        from prompt_rover import ConceptTransformationVisualizer
        
        print("\nTest creazione visualizzatore...")
        visualizer = ConceptTransformationVisualizer()
        print("✓ Visualizzatore creato con successo")
        
        print("\nTest estrazione concetti...")
        test_text = "I fiori rappresentano la bellezza e la saggezza della natura."
        concepts = visualizer.extract_concepts_alternative(test_text, is_user=True)
        print(f"✓ Estratti {len(concepts)} concetti")
        
        print("\nTest embeddings...")
        concepts_with_embeddings = visualizer.compute_embeddings(concepts)
        print("✓ Embeddings calcolati")
        
        print("\nTest costruzione grafo...")
        graph = visualizer.build_concept_graph(concepts_with_embeddings)
        print(f"✓ Grafo costruito con {graph.number_of_nodes()} nodi e {graph.number_of_edges()} archi")
        
        print("\nTest riduzione dimensionale...")
        df = visualizer.reduce_dimensions(concepts_with_embeddings, method="pca")
        print(f"✓ Dimensioni ridotte, DataFrame con {len(df)} righe")
        
        print("\n✅ Test funzionalità di base completati!")
        return True
        
    except Exception as e:
        print(f"\n❌ Errore nei test funzionali: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chat_mode():
    """Test modalità chat"""
    try:
        from prompt_rover import ConceptTransformationVisualizer
        
        print("\nTest modalità chat...")
        visualizer = ConceptTransformationVisualizer()
        
        print("Inizializzazione chat...")
        visualizer.initialize_chat_mode()
        print("✓ Chat inizializzata")
        
        print("Elaborazione messaggio...")
        df, fig, status = visualizer.process_new_message(
            "Ciao, parliamo di intelligenza artificiale",
            is_user=True,
            use_llm=False
        )
        print(f"✓ Messaggio elaborato: {status}")
        
        print("\n✅ Test modalità chat completati!")
        return True
        
    except Exception as e:
        print(f"\n❌ Errore nei test chat: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Test Prompt Rover ===\n")
    
    # Esegui i test
    import_ok = test_imports()
    
    if import_ok:
        print("\n" + "="*40 + "\n")
        func_ok = test_basic_functionality()
        
        print("\n" + "="*40 + "\n") 
        chat_ok = test_chat_mode()
        
        print("\n" + "="*40 + "\n")
        if func_ok and chat_ok:
            print("✅ TUTTI I TEST SONO PASSATI!")
        else:
            print("❌ Alcuni test sono falliti")
    else:
        print("\n❌ Import falliti, impossibile eseguire altri test") 