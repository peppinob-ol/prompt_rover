"""
Gestione della cache per Prompt Rover
"""

import hashlib
from collections import OrderedDict
from ..config import CACHE_ENABLED, MAX_CACHE_SIZE
from .logging import get_logger

cache_logger = get_logger('cache')

class CacheManager:
    """
    Gestore della cache con politica LRU (Least Recently Used)
    """
    
    def __init__(self, max_size=MAX_CACHE_SIZE, enabled=CACHE_ENABLED):
        """
        Inizializza il gestore della cache
        
        Args:
            max_size: Dimensione massima della cache
            enabled: Se la cache è abilitata
        """
        self.max_size = max_size
        self.enabled = enabled
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        cache_logger.info(f"Cache inizializzata (max_size={max_size}, enabled={enabled})")
    
    def _generate_key(self, text, params=None):
        """
        Genera una chiave univoca basata su testo e parametri
        
        Args:
            text: Testo da cui generare la chiave
            params: Dizionario di parametri aggiuntivi
            
        Returns:
            Chiave string per la cache
        """
        # Genera hash dal testo
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Se non ci sono parametri, usa solo l'hash del testo
        if not params:
            return text_hash
        
        # Altrimenti, incorpora i parametri nella chiave
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        param_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()[:8]
        
        return f"{text_hash}_{param_hash}"
    
    def get(self, key):
        """
        Recupera un valore dalla cache
        
        Args:
            key: Chiave da cercare
            
        Returns:
            Valore se trovato, None altrimenti
        """
        if not self.enabled:
            return None
            
        if key in self.cache:
            # Sposta l'elemento alla fine (più recente)
            self.cache.move_to_end(key)
            self.hits += 1
            cache_logger.debug(f"Cache hit: {key}")
            return self.cache[key]
        
        self.misses += 1
        cache_logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, key, value):
        """
        Salva un valore nella cache
        
        Args:
            key: Chiave
            value: Valore da salvare
        """
        if not self.enabled:
            return
            
        # Se la chiave esiste già, spostala alla fine
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            # Se la cache è piena, rimuovi l'elemento più vecchio
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                cache_logger.debug(f"Cache eviction: {oldest_key}")
            
        self.cache[key] = value
        cache_logger.debug(f"Cache set: {key}")
    
    def clear(self):
        """Svuota la cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        cache_logger.info("Cache svuotata")
    
    def get_stats(self):
        """
        Ottiene le statistiche della cache
        
        Returns:
            Dizionario con le statistiche
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "enabled": self.enabled
        }
    
    def __len__(self):
        return len(self.cache)
    
    def __contains__(self, key):
        return key in self.cache 