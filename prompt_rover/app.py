"""
Prompt Rover - HuggingFace Spaces App
Entry point per il deployment su HF Spaces
"""

import os
import sys

# Aggiungi il percorso della directory padre al path Python (per trovare prompt_rover/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importa e lancia l'app Gradio
from prompt_rover.ui import create_gradio_interface

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