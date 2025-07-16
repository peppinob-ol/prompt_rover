"""
Classe principale per la visualizzazione delle trasformazioni concettuali
"""

import pandas as pd
import matplotlib.pyplot as plt
import traceback
from typing import Optional, Tuple, Dict

from ..config import DEFAULT_EMBEDDING_MODEL, OPENAI_API_KEY
from ..utils import get_logger, setup_logging, CacheManager
from ..visualization import InteractiveVisualizer
from ..chat import ChatHandler
from .concept_extractor import ConceptExtractor
from .embeddings import EmbeddingManager
from .graph_builder import GraphBuilder
from .dimension_reducer import DimensionReducer

# Configura logging all'importazione
setup_logging()

logger = get_logger('performance')

class ConceptTransformationVisualizer:
    """
    Visualizzatore principale per le trasformazioni concettuali
    """
    
    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL, 
                 openai_api_key: Optional[str] = None):
        """
        Inizializza il visualizzatore
        
        Args:
            embedding_model: Modello SentenceTransformer da usare
            openai_api_key: Chiave API OpenAI (opzionale)
        """
        logger.info("Inizializzazione ConceptTransformationVisualizer...")
        
        # Usa la chiave API fornita o quella dalle variabili d'ambiente
        api_key = openai_api_key or OPENAI_API_KEY
        
        # Inizializza componenti
        self.concept_extractor = ConceptExtractor(api_key)
        self.embedding_manager = EmbeddingManager(embedding_model)
        self.graph_builder = GraphBuilder()
        self.dimension_reducer = DimensionReducer()
        self.visualizer = InteractiveVisualizer()
        
        # Inizializza chat handler
        self.chat_handler = ChatHandler(self.concept_extractor.llm_client)
        
        # Cache per ottimizzazione
        self.cache = CacheManager()
        
        # Mantieni compatibilità con vecchio codice
        self.llm_client = self.concept_extractor.llm_client
        self.embedding_model = self.embedding_manager.model
        self.concept_graph = None
        
        logger.info("Inizializzazione completata")
    
    def initialize_openai_client(self, api_key: str) -> bool:
        """
        Inizializza il client OpenAI
        
        Args:
            api_key: Chiave API
            
        Returns:
            True se successo
        """
        return self.concept_extractor._initialize_openai(api_key)
    
    def initialize_chat_mode(self):
        """Inizializza la modalità chat"""
        self.chat_handler.reset()
        self.dimension_reducer.reset()
        self.cache.clear()
        return "Modalità chat inizializzata"
    
    def extract_concepts_with_llm(self, text: str, is_user: bool, 
                                 model: str = "gpt-4", timeout: int = 30):
        """Estrae concetti usando LLM (wrapper per compatibilità)"""
        return self.concept_extractor.extract_concepts(
            text, is_user, use_llm=True, model=model
        )
    
    def extract_concepts_alternative(self, text: str, is_user: bool):
        """Estrae concetti senza LLM (wrapper per compatibilità)"""
        return self.concept_extractor.extract_concepts(
            text, is_user, use_llm=False
        )
    
    def compute_embeddings(self, concepts, name_weight: float = 1.0):
        """Calcola embeddings (wrapper per compatibilità)"""
        return self.embedding_manager.compute_embeddings(concepts, name_weight)
    
    def build_concept_graph(self, concepts, k: int = 3, use_mst: bool = True,
                           show_io_links: bool = False):
        """Costruisce grafo (wrapper per compatibilità)"""
        graph = self.graph_builder.build_graph(
            concepts, k, use_mst, show_io_links
        )
        self.concept_graph = graph
        return graph
    
    def reduce_dimensions(self, concepts, method: str = "umap",
                         n_neighbors: int = 5, min_dist: float = 0.1,
                         perplexity: int = 30):
        """Riduce dimensioni (wrapper per compatibilità)"""
        return self.dimension_reducer.reduce_dimensions(
            concepts, method, n_neighbors, min_dist, perplexity
        )
    
    def visualize_concepts_interactive(self, df: pd.DataFrame,
                                     show_evolution: bool = False,
                                     title: Optional[str] = None):
        """Crea visualizzazione interattiva"""
        return self.visualizer.visualize_concepts(
            df, self.concept_graph, show_evolution, title
        )
    
    def process_text_pair(self, input_text: str, output_text: str,
                         use_llm: bool = False, name_weight: float = 1.0,
                         dim_reduction: str = "umap", use_mst: bool = True,
                         show_io_links: bool = False) -> Tuple:
        """
        Processa una coppia di testi e crea visualizzazioni
        
        Args:
            input_text: Testo di input
            output_text: Testo di output
            use_llm: Se usare LLM per estrazione concetti
            name_weight: Peso nome vs descrizione
            dim_reduction: Metodo riduzione dimensionale
            use_mst: Se usare MST per il grafo
            show_io_links: Se mostrare collegamenti input-output
            
        Returns:
            Tupla (DataFrame, Figura, Messaggio di stato)
        """
        logger.info(f"Elaborazione coppia di testi - Input: {len(input_text)} caratteri, Output: {len(output_text)} caratteri")
        
        try:
            # Genera chiavi cache
            cache_params = {
                "use_llm": use_llm,
                "name_weight": name_weight,
                "dim_reduction": dim_reduction
            }
            
            # Estrai concetti con cache
            input_concepts = self._extract_cached_concepts(
                input_text, True, use_llm, name_weight, "input"
            )
            
            output_concepts = self._extract_cached_concepts(
                output_text, False, use_llm, name_weight, "output"
            )
            
            # Combina concetti
            all_concepts = input_concepts + output_concepts
            
            # Costruisci grafo
            self.build_concept_graph(all_concepts, use_mst=use_mst,
                                   show_io_links=show_io_links)
            
            # Riduci dimensionalità
            df = self.reduce_dimensions(all_concepts, method=dim_reduction)
            
            # Crea visualizzazione
            fig = self.visualize_concepts_interactive(df)
            
            return df, fig, "Elaborazione completata con successo"
            
        except Exception as e:
            logger.error(f"Errore nell'elaborazione: {e}")
            traceback.print_exc()
            
            # Restituisci risultati vuoti in caso di errore
            empty_df = pd.DataFrame(columns=["label", "category", "description", "source", "x", "y"])
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Errore: {str(e)}", ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            
            return empty_df, fig, f"Errore: {str(e)}"
    
    def process_new_message(self, message: str, is_user: bool = True,
                           name_weight: float = 1.0,
                           dim_reduction_method: str = "umap",
                           use_mst: bool = True,
                           show_io_links: bool = False,
                           use_llm: bool = False) -> Tuple:
        """
        Processa un nuovo messaggio in modalità chat
        
        Args:
            message: Testo del messaggio
            is_user: Se il messaggio è dell'utente
            name_weight: Peso nome vs descrizione
            dim_reduction_method: Metodo riduzione dimensionale
            use_mst: Se usare MST
            show_io_links: Se mostrare collegamenti
            use_llm: Se usare LLM
            
        Returns:
            Tupla (DataFrame, Figura, Messaggio di stato)
        """
        content_type = "user" if is_user else "assistant"
        logger.info(f"Elaborazione nuovo messaggio - {content_type} - Lunghezza: {len(message)} caratteri")
        
        try:
            # Aggiungi messaggio
            msg_info = self.chat_handler.add_message(message, is_user)
            
            # Estrai concetti
            new_concepts = self._extract_cached_concepts(
                message, is_user, use_llm, name_weight, content_type
            )
            
            # Aggiorna concetti nella chat
            self.chat_handler.update_concepts(
                new_concepts, msg_info["id"], content_type
            )
            
            # Verifica limite
            if self.chat_handler.check_concept_limit():
                return None, None, "Limite concetti raggiunto, resetta la conversazione"
            
            # Ricostruisci grafo con tutti i concetti
            self.build_concept_graph(
                self.chat_handler.chat_concepts,
                use_mst=use_mst,
                show_io_links=show_io_links
            )
            
            # Riduci dimensionalità
            df = self.reduce_dimensions(
                self.chat_handler.chat_concepts,
                method=dim_reduction_method
            )
            
            # Crea visualizzazione con evoluzione
            fig = self.visualize_concepts_interactive(df, show_evolution=True)
            
            return df, fig, f"Messaggio elaborato: trovati {len(new_concepts)} nuovi concetti"
            
        except Exception as e:
            logger.error(f"Errore nell'elaborazione messaggio: {e}", exc_info=True)
            
            # Restituisci risultati vuoti
            empty_df = pd.DataFrame(columns=["label", "category", "description", "source", "x", "y"])
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Errore: {str(e)}", ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            
            return empty_df, fig, f"Errore: {str(e)}"
    
    def generate_chat_response(self, history):
        """Genera risposta chat (wrapper per compatibilità)"""
        # Converti history nel formato atteso dal chat handler
        for msg in history:
            if msg not in [h["content"] for h in self.chat_handler.chat_history]:
                is_user = msg.get("role") == "user"
                self.chat_handler.add_message(msg["content"], is_user)
        
        return self.chat_handler.generate_response()
    
    def _extract_cached_concepts(self, text: str, is_user: bool,
                               use_llm: bool, name_weight: float,
                               content_type: str):
        """
        Estrae concetti con cache
        
        Args:
            text: Testo da analizzare
            is_user: Se è testo utente
            use_llm: Se usare LLM
            name_weight: Peso nome/descrizione
            content_type: Tipo contenuto per source
            
        Returns:
            Lista di concetti con embeddings
        """
        # Genera chiave cache
        cache_key = self.cache._generate_key(text, {
            "use_llm": use_llm,
            "name_weight": name_weight
        })
        
        # Verifica cache
        cached_concepts = self.cache.get(cache_key)
        if cached_concepts is not None:
            logger.info(f"Uso concetti dalla cache per {content_type}")
            # Aggiorna source per compatibilità
            for c in cached_concepts:
                c["source"] = content_type
            return cached_concepts
        
        # Estrai concetti
        concepts = self.concept_extractor.extract_concepts(
            text, is_user, use_llm
        )
        
        # Aggiungi source per compatibilità con input/output mode
        for c in concepts:
            c["source"] = content_type
        
        # Calcola embeddings
        concepts = self.embedding_manager.compute_embeddings(
            concepts, name_weight
        )
        
        # Salva in cache
        self.cache.set(cache_key, concepts)
        
        return concepts
    
    # Metodi per compatibilità con vecchio codice
    @property
    def chat_concepts(self):
        return self.chat_handler.chat_concepts
    
    @property
    def chat_history(self):
        return self.chat_handler.chat_history
    
    @property
    def message_counter(self):
        return self.chat_handler.message_counter 