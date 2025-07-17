"""
Interfaccia Gradio per Prompt Rover
"""

import gradio as gr
import traceback
import os
from urllib.parse import parse_qs, urlparse

from ..core import ConceptTransformationVisualizer
from ..config import OPENAI_API_KEY, GRADIO_THEME, GRADIO_SHARE, GRADIO_DEBUG
from ..utils import get_logger

logger = get_logger('ui')

def create_gradio_interface(visualizer=None):
    """
    Crea l'interfaccia Gradio
    
    Args:
        visualizer: Istanza di ConceptTransformationVisualizer (opzionale)
        
    Returns:
        Interfaccia Gradio
    """
    # Crea visualizzatore se non fornito
    if visualizer is None:
        visualizer = ConceptTransformationVisualizer()
        visualizer.initialize_chat_mode()  # Inizializza chat all'avvio
    
    # Carica chiave API da variabili d'ambiente se disponibile
    default_api_key = OPENAI_API_KEY

    with gr.Blocks(title="Prompt Rover", theme=GRADIO_THEME) as demo:
        gr.Markdown("# Prompt Rover")
        
        # Layout principale orizzontale
        with gr.Row():
            # Colonna sinistra per input con tabs
            with gr.Column(scale=1):
                with gr.Tabs() as tabs:
                    # Tab per modalità Input/Output
                    with gr.TabItem("Modalità Input/Output"):
                        input_text = gr.Textbox(
                            label="Testo Input", 
                            placeholder="Inserisci il testo di input",
                            lines=5,
                            elem_id="input_text"
                        )
                        output_text = gr.Textbox(
                            label="Testo Output",
                            placeholder="Inserisci il testo di output",
                            lines=10,
                            elem_id="output_text"
                        )
                        
                        with gr.Accordion("Opzioni avanzate", open=False):
                            name_weight_io = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                                label="Peso del nome vs. descrizione"
                            )
                            dim_reduction_io = gr.Radio(
                                ["umap", "tsne", "pca"],
                                label="Metodo di riduzione dimensionale",
                                value="tsne"
                            )
                            use_mst_io = gr.Checkbox(
                                label="Usa Minimum Spanning Tree",
                                value=True
                            )
                            show_io_links_io = gr.Checkbox(
                                label="Mostra connessioni input-output",
                                value=True
                            )
                            use_llm_io = gr.Checkbox(
                                label="Usa LLM per estrazione concetti",
                                value=True
                            )
                            openai_api_key_io = gr.Textbox(
                                label="Chiave API OpenAI (opzionale)",
                                value=default_api_key,
                                placeholder="Inserisci se vuoi usare LLM",
                                type="password"
                            )
                        
                        submit_btn = gr.Button("Elabora e visualizza", variant="primary")
                        example_btn = gr.Button("Carica esempio fiori e saggezza")
                    
                    # Tab per modalità Chat
                    with gr.TabItem("Modalità Chat"):
                        # Compatibilità con diverse versioni di Gradio
                        import inspect as _ins
                        _chatbot_kwargs = {
                            "height": 400,
                            "label": "Conversazione",
                        }
                        if "type" in _ins.signature(gr.Chatbot).parameters:
                            _chatbot_kwargs["type"] = "messages"
                        chatbot = gr.Chatbot(**_chatbot_kwargs)
                        chat_msg = gr.Textbox(
                            placeholder="Scrivi un messaggio qui...",
                            label="Messaggio"
                        )
                        
                        with gr.Accordion("Opzioni avanzate", open=False):
                            name_weight_chat = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                                label="Peso del nome vs. descrizione"
                            )
                            dim_reduction_chat = gr.Radio(
                                ["umap", "tsne", "pca"],
                                label="Metodo di riduzione dimensionale",
                                value="tsne"
                            )
                            use_mst_chat = gr.Checkbox(
                                label="Usa Minimum Spanning Tree",
                                value=True
                            )
                            show_io_links_chat = gr.Checkbox(
                                label="Mostra connessioni input-output",
                                value=False
                            )
                            use_llm_chat = gr.Checkbox(
                                label="Usa LLM per estrazione concetti",
                                value=True
                            )
                            openai_api_key_chat = gr.Textbox(
                                label="Chiave API OpenAI (opzionale)",
                                value=default_api_key,
                                placeholder="Inserisci se vuoi usare LLM",
                                type="password"
                            )
                        
                        with gr.Row():
                            chat_submit = gr.Button("Invia", variant="primary")
                            chat_clear = gr.Button("Cancella conversazione")
                            reinit_btn = gr.Button("Reinizializza sistema")
            
            # Colonna destra per visualizzazione e risultati
            with gr.Column(scale=2):
                static_output = gr.Plot(label="Visualizzazione concetti")
                df_output = gr.DataFrame(label="Concetti estratti")
                status = gr.Textbox(
                    label="Stato",
                    interactive=False,
                    value="Sistema inizializzato e pronto all'uso"
                )
        
        # Funzioni per gestire l'interfaccia
        
        def reinit_chat():
            """Reinizializza la modalità chat"""
            return visualizer.initialize_chat_mode()
        
        def process_chat_message(message, history, name_weight, dim_reduction,
                               use_mst, show_io_links, use_llm, openai_api_key):
            """Processa un messaggio chat"""
            if not message.strip():
                return "", history, None, None, "Messaggio vuoto"
            
            # Configura chiave API se fornita e richiesta
            if use_llm and openai_api_key.strip():
                if not visualizer.initialize_openai_client(openai_api_key):
                    return "", history, None, None, "Errore nell'inizializzazione API OpenAI"
            
            try:
                if history is None:
                    history = []
                
                # Formato messaggi
                history.append({"role": "user", "content": message})
                
                # Analizza messaggio utente
                df, fig_static, status_msg = visualizer.process_new_message(
                    message, is_user=True,
                    name_weight=name_weight,
                    dim_reduction_method=dim_reduction,
                    use_mst=use_mst,
                    show_io_links=show_io_links,
                    use_llm=use_llm
                )
                
                # Genera risposta assistente
                assistant_response = visualizer.generate_chat_response(history)
                history.append({"role": "assistant", "content": assistant_response})
                
                # Analizza anche la risposta del sistema
                df, fig_static, status_msg = visualizer.process_new_message(
                    assistant_response, is_user=False,
                    name_weight=name_weight,
                    dim_reduction_method=dim_reduction,
                    use_mst=use_mst,
                    show_io_links=show_io_links,
                    use_llm=use_llm
                )
                
                return "", history, df, fig_static, status_msg
                
            except Exception as e:
                error_msg = f"Errore durante l'elaborazione: {str(e)}"
                logger.error(error_msg)
                traceback.print_exc()
                
                if history is None:
                    history = []
                history.append({
                    "role": "assistant",
                    "content": "Mi dispiace, si è verificato un errore durante l'elaborazione"
                })
                
                return "", history, None, None, error_msg
        
        def clear_chat():
            """Cancella la conversazione"""
            visualizer.initialize_chat_mode()
            return "", [], None, None, "Conversazione cancellata"
        
        def load_example():
            """Carica esempio predefinito"""
            input_text_example = "fiori saggezza"
            output_text_example = """Se pensiamo al simbolismo dei fiori in relazione alla saggezza, il fiore di loto è uno dei più rappresentativi. Nell'immaginario di molte culture, soprattutto orientali, il loto cresce nel fango ma emerge puro e immacolato sulla superficie dell'acqua, simboleggiando la capacità di elevazione spirituale e intellettuale, proprio come chi acquisisce saggezza attraverso le difficoltà della vita.

Il salvia ha anche un nome che richiama la saggezza, derivando dal latino salvus (sano, salvo). Questa pianta, oltre alle sue proprietà curative, è spesso associata alla prudenza e all'esperienza.

Se pensiamo invece alla tradizione occidentale, la quercia, pur essendo un albero e non un fiore, è simbolo di saggezza per la sua longevità e la sua capacità di resistere alle tempeste, proprio come una mente saggia che sa affrontare le avversità senza spezzarsi."""
            
            return input_text_example, output_text_example
        
        def process_with_openai_key(input_text, output_text, use_llm, name_weight,
                                   dim_reduction, use_mst, show_io_links, openai_api_key):
            """Processa testo con inizializzazione chiave API"""
            # Se l'utente vuole usare LLM, inizializza il client
            if use_llm and openai_api_key.strip():
                visualizer.initialize_openai_client(openai_api_key)
            
            # Procedi con l'elaborazione normale
            return visualizer.process_text_pair(
                input_text, output_text, use_llm,
                name_weight, dim_reduction,
                use_mst, show_io_links
            )
        
        # Collegamenti per modalità Input/Output
        submit_btn.click(
            process_with_openai_key,
            inputs=[
                input_text, output_text, use_llm_io, name_weight_io,
                dim_reduction_io, use_mst_io, show_io_links_io, openai_api_key_io
            ],
            outputs=[df_output, static_output, status]
        )
        
        example_btn.click(
            load_example,
            outputs=[input_text, output_text]
        )
        
        # Collegamenti per modalità Chat
        chat_msg.submit(
            process_chat_message,
            inputs=[
                chat_msg, chatbot, name_weight_chat, dim_reduction_chat,
                use_mst_chat, show_io_links_chat, use_llm_chat, openai_api_key_chat
            ],
            outputs=[chat_msg, chatbot, df_output, static_output, status]
        )
        
        chat_submit.click(
            process_chat_message,
            inputs=[
                chat_msg, chatbot, name_weight_chat, dim_reduction_chat,
                use_mst_chat, show_io_links_chat, use_llm_chat, openai_api_key_chat
            ],
            outputs=[chat_msg, chatbot, df_output, static_output, status]
        )
        
        chat_clear.click(
            clear_chat,
            outputs=[chat_msg, chatbot, df_output, static_output, status]
        )
        
        reinit_btn.click(
            reinit_chat,
            outputs=[status]
        )
        
        # Funzione per gestire parametri URL
        def handle_url_params():
            """
            Gestisce parametri URL per pre-popolare input/output
            
            Supporta i seguenti parametri:
            - input: testo da mettere nel campo input
            - output: testo da mettere nel campo output
            - autorun: se "true", esegue automaticamente l'elaborazione
            
            Esempio URL:
            https://huggingface.co/spaces/user/prompt-rover?input=ciao&output=hello&autorun=true
            """
            logger.info("Controllo parametri URL...")
            
            # Valori da restituire
            new_input = input_text.value
            new_output = output_text.value
            should_process = False
            status_msg = "Sistema pronto"
            
            try:
                # Per HuggingFace Spaces, usa JavaScript per ottenere l'URL
                js_code = """
                () => {
                    const urlParams = new URLSearchParams(window.location.search);
                    return {
                        input: urlParams.get('input') || '',
                        output: urlParams.get('output') || '',
                        autorun: urlParams.get('autorun') || 'false'
                    };
                }
                """
                
                # Ottieni parametri via JavaScript
                params = gr.Interface.load(js_code)
                
                if params:
                    # Aggiorna input se presente
                    if params.get('input'):
                        new_input = params['input']
                        logger.info(f"Input da URL: {new_input[:50]}...")
                    
                    # Aggiorna output se presente
                    if params.get('output'):
                        new_output = params['output']
                        logger.info(f"Output da URL: {new_output[:50]}...")
                    
                    # Controlla se eseguire automaticamente
                    if params.get('autorun', '').lower() == 'true':
                        should_process = True
                        status_msg = "Elaborazione automatica da URL..."
                    elif params.get('input') or params.get('output'):
                        status_msg = "Campi popolati da URL. Clicca 'Elabora' per visualizzare."
                
            except Exception as e:
                logger.warning(f"Impossibile leggere parametri URL: {e}")
                # Fallback: prova a leggere da variabili d'ambiente (per test locali)
                try:
                    import urllib.parse
                    query_string = os.environ.get('QUERY_STRING', '')
                    if query_string:
                        params = urllib.parse.parse_qs(query_string)
                        if 'input' in params:
                            new_input = params['input'][0]
                        if 'output' in params:
                            new_output = params['output'][0]
                        if params.get('autorun', ['false'])[0].lower() == 'true':
                            should_process = True
                except:
                    pass
            
            return new_input, new_output, status_msg
        
        # Aggiungi JavaScript per gestire parametri URL in HF Spaces
        demo.load(
            fn=None,
            inputs=None,
            outputs=None,
            js="""
            () => {
                // Leggi parametri URL
                const urlParams = new URLSearchParams(window.location.search);
                const inputParam = urlParams.get('input');
                const outputParam = urlParams.get('output');
                const autorun = urlParams.get('autorun');
                
                // Popola campi se ci sono parametri
                if (inputParam) {
                    const inputField = document.querySelector('#input_text textarea');
                    if (inputField) {
                        inputField.value = decodeURIComponent(inputParam);
                        inputField.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                }
                
                if (outputParam) {
                    const outputField = document.querySelector('#output_text textarea');
                    if (outputField) {
                        outputField.value = decodeURIComponent(outputParam);
                        outputField.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                }
                
                // Se autorun=true, clicca automaticamente il bottone
                if (autorun === 'true') {
                    setTimeout(() => {
                        const submitBtn = document.querySelector('button[variant="primary"]');
                        if (submitBtn && submitBtn.textContent.includes('Elabora')) {
                            submitBtn.click();
                        }
                    }, 1000);
                }
                
                console.log('URL params processed:', { input: inputParam, output: outputParam, autorun });
            }
            """
        )
    
    return demo 