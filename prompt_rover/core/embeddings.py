"""
Modulo per la gestione degli embeddings dei concetti
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

from ..config import DEFAULT_EMBEDDING_MODEL, LABEL, DESCRIPTION, EMBEDDING
from ..utils import get_logger, log_execution_time, CacheManager

logger = get_logger('embeddings')

class EmbeddingManager:
    """
    Gestisce il calcolo e la cache degli embeddings
    """
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """
        Inizializza il gestore degli embeddings
        
        Args:
            model_name: Nome del modello SentenceTransformer da usare
        """
        logger.info(f"Caricamento modello di embedding: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Modello {model_name} caricato con successo")
        except Exception as e:
            logger.error(f"Errore nel caricamento del modello: {str(e)}")
            raise
            
        # Cache per gli embeddings
        self.cache = CacheManager()
        
        # Storage separato per nomi e descrizioni
        self.name_embeddings = {}
        self.description_embeddings = {}
        
    @log_execution_time
    def compute_embeddings(self, concepts: List[Dict], name_weight: float = 1.0) -> List[Dict]:
        """
        Calcola gli embeddings per una lista di concetti
        
        Args:
            concepts: Lista di concetti con metadata
            name_weight: Peso da dare al nome vs. descrizione (0.0-1.0)
            
        Returns:
            La stessa lista con embeddings aggiunti
        """
        logger.info(f"Calcolo embeddings per {len(concepts)} concetti")
        
        if len(concepts) == 0:
            return concepts
            
        # Estrai etichette e descrizioni
        labels = [concept[LABEL] for concept in concepts]
        descriptions = [concept.get(DESCRIPTION, concept[LABEL]) for concept in concepts]
        
        # Calcola embeddings separati per nomi e descrizioni
        name_embeddings = self._encode_batch(labels, prefix="name")
        desc_embeddings = self._encode_batch(descriptions, prefix="desc")
        
        # Calcola embeddings pesati
        for i, concept in enumerate(concepts):
            # Salva embeddings separati per riferimento futuro
            self.name_embeddings[concept[LABEL]] = name_embeddings[i]
            self.description_embeddings[concept[LABEL]] = desc_embeddings[i]
            
            # Calcola embedding pesato
            weighted_emb = self._compute_weighted_embedding(
                name_embeddings[i], 
                desc_embeddings[i], 
                name_weight
            )
            
            # Aggiungi al concetto
            concept[EMBEDDING] = weighted_emb
            
        return concepts
    
    def _encode_batch(self, texts: List[str], prefix: str = "") -> np.ndarray:
        """
        Codifica un batch di testi, usando la cache quando possibile
        
        Args:
            texts: Lista di testi da codificare
            prefix: Prefisso per la chiave della cache
            
        Returns:
            Array numpy con gli embeddings
        """
        embeddings = []
        texts_to_encode = []
        text_indices = []
        
        # Controlla quali testi sono già in cache
        for i, text in enumerate(texts):
            cache_key = f"{prefix}:{text}"
            cached_embedding = self.cache.get(cache_key)
            
            if cached_embedding is not None:
                embeddings.append((i, cached_embedding))
            else:
                texts_to_encode.append(text)
                text_indices.append(i)
        
        # Codifica i testi non in cache
        if texts_to_encode:
            logger.info(f"Codifica {len(texts_to_encode)} testi non in cache")
            new_embeddings = self.model.encode(texts_to_encode)
            
            # Salva in cache e aggiungi alla lista
            for idx, text, embedding in zip(text_indices, texts_to_encode, new_embeddings):
                cache_key = f"{prefix}:{text}"
                self.cache.set(cache_key, embedding)
                embeddings.append((idx, embedding))
        
        # Riordina embeddings secondo l'ordine originale
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
    
    def _compute_weighted_embedding(self, name_emb: np.ndarray, 
                                  desc_emb: np.ndarray, 
                                  name_weight: float) -> np.ndarray:
        """
        Calcola un embedding pesato tra nome e descrizione
        
        Args:
            name_emb: Embedding del nome
            desc_emb: Embedding della descrizione
            name_weight: Peso del nome (0.0-1.0)
            
        Returns:
            Embedding pesato e normalizzato
        """
        # Calcola combinazione pesata
        weighted_emb = name_weight * name_emb + (1 - name_weight) * desc_emb
        
        # Normalizza
        norm = np.linalg.norm(weighted_emb)
        if norm > 0:
            weighted_emb = weighted_emb / norm
            
        return weighted_emb
    
    def get_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calcola la similarità coseno tra due embeddings
        
        Args:
            emb1: Primo embedding
            emb2: Secondo embedding
            
        Returns:
            Similarità coseno (0-1)
        """
        # Normalizza embeddings
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        emb1_norm = emb1 / norm1
        emb2_norm = emb2 / norm2
        
        # Calcola similarità coseno
        return float(np.dot(emb1_norm, emb2_norm))
    
    def find_similar_concepts(self, target_embedding: np.ndarray, 
                            concept_embeddings: Dict[str, np.ndarray], 
                            top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Trova i concetti più simili a un embedding target
        
        Args:
            target_embedding: Embedding di riferimento
            concept_embeddings: Dizionario label -> embedding
            top_k: Numero di concetti simili da restituire
            
        Returns:
            Lista di tuple (label, similarity) ordinate per similarità
        """
        similarities = []
        
        for label, embedding in concept_embeddings.items():
            sim = self.get_similarity(target_embedding, embedding)
            similarities.append((label, sim))
        
        # Ordina per similarità decrescente
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def clear_cache(self):
        """Svuota la cache degli embeddings"""
        self.cache.clear()
        self.name_embeddings.clear()
        self.description_embeddings.clear()
        logger.info("Cache embeddings svuotata") 