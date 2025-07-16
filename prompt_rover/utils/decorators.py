"""
Decoratori per Prompt Rover
"""

import time
from functools import wraps
from .logging import get_logger

perf_logger = get_logger('performance')

def log_execution_time(func):
    """
    Decoratore per misurare il tempo di esecuzione delle funzioni
    
    Args:
        func: Funzione da decorare
        
    Returns:
        Funzione decorata
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Log del tempo di esecuzione
        perf_logger.info(f"{func.__name__} eseguito in {execution_time:.4f} secondi")

        # Se l'esecuzione è lenta (> 1 secondo), log come warning
        if execution_time > 1.0:
            perf_logger.warning(f"Potenziale bottleneck: {func.__name__} ha richiesto {execution_time:.4f} secondi")

        return result
    return wrapper

def cached(cache_key_func=None):
    """
    Decoratore per cacheare i risultati delle funzioni
    
    Args:
        cache_key_func: Funzione per generare la chiave di cache
        
    Returns:
        Decoratore
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, '_cache') and self._cache is not None:
                # Genera la chiave di cache
                if cache_key_func:
                    cache_key = cache_key_func(self, *args, **kwargs)
                else:
                    cache_key = str(args) + str(kwargs)
                
                # Controlla se il risultato è in cache
                if cache_key in self._cache:
                    return self._cache[cache_key]
                
                # Calcola il risultato
                result = func(self, *args, **kwargs)
                
                # Salva in cache
                self._cache[cache_key] = result
                return result
            else:
                # Se non c'è cache, esegui normalmente
                return func(self, *args, **kwargs)
        return wrapper
    return decorator 