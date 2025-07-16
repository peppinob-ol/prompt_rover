"""
Prompt Rover - HuggingFace Spaces App
Entry point per il deployment su HF Spaces
"""

import os
import sys

# Configurazione path per funzionare sia in locale che su HF Spaces
current_dir = os.path.dirname(os.path.abspath(__file__))

# Se siamo in prompt_rover/ (locale), aggiungi parent directory
if os.path.basename(current_dir) == 'prompt_rover':
    sys.path.insert(0, os.path.dirname(current_dir))
    from prompt_rover.ui import create_gradio_interface
else:
    # Se siamo su HF Spaces (root), importa direttamente
    sys.path.insert(0, current_dir)
    from ui import create_gradio_interface

if __name__ == "__main__":
    # Crea e lancia l'interfaccia
    demo = create_gradio_interface()
    
    # Configurazione per HF Spaces
    demo.launch(
        share=False,  # Non condividere pubblicamente su Gradio
        debug=True,   # Debug attivo per vedere errori
        server_name="0.0.0.0",  # Ascolta su tutte le interfacce
        server_port=7860  # Porta standard HF Spaces
    ) 