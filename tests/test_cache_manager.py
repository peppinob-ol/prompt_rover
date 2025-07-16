"""
Test per il modulo CacheManager
"""

import pytest
from prompt_rover.utils import CacheManager

class TestCacheManager:
    """Test per CacheManager"""
    
    def test_initialization(self):
        """Test inizializzazione"""
        cache = CacheManager(max_size=10)
        assert cache.max_size == 10
        assert cache.enabled == True
        assert len(cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_set_and_get(self):
        """Test set e get di base"""
        cache = CacheManager()
        
        # Set value
        cache.set("key1", "value1")
        assert len(cache) == 1
        
        # Get value
        value = cache.get("key1")
        assert value == "value1"
        assert cache.hits == 1
        assert cache.misses == 0
        
        # Get non-existent key
        value = cache.get("key2")
        assert value is None
        assert cache.misses == 1
    
    def test_cache_disabled(self):
        """Test con cache disabilitata"""
        cache = CacheManager(enabled=False)
        
        # Set non dovrebbe salvare
        cache.set("key1", "value1")
        assert len(cache) == 0
        
        # Get dovrebbe sempre restituire None
        value = cache.get("key1")
        assert value is None
    
    def test_lru_eviction(self):
        """Test evizione LRU"""
        cache = CacheManager(max_size=3)
        
        # Riempi la cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        assert len(cache) == 3
        
        # Aggiungi un quarto elemento - dovrebbe rimuovere key1
        cache.set("key4", "value4")
        assert len(cache) == 3
        assert cache.get("key1") is None  # Rimosso
        assert cache.get("key4") == "value4"  # Aggiunto
    
    def test_lru_ordering(self):
        """Test che LRU mantenga l'ordine corretto"""
        cache = CacheManager(max_size=3)
        
        # Riempi cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Accedi a key1 per renderlo più recente
        cache.get("key1")
        
        # Aggiungi key4 - dovrebbe rimuovere key2 (meno recente)
        cache.set("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Ancora presente
        assert cache.get("key2") is None      # Rimosso
        assert cache.get("key3") == "value3"  # Ancora presente
        assert cache.get("key4") == "value4"  # Aggiunto
    
    def test_generate_key(self):
        """Test generazione chiavi"""
        cache = CacheManager()
        
        # Solo testo
        key1 = cache._generate_key("test text", None)
        key2 = cache._generate_key("test text", None)
        assert key1 == key2  # Stesso testo = stessa chiave
        
        # Testo diverso
        key3 = cache._generate_key("different text", None)
        assert key1 != key3
        
        # Con parametri
        key4 = cache._generate_key("test text", {"param": "value"})
        assert key1 != key4  # Parametri diversi = chiave diversa
        
        # Stessi parametri
        key5 = cache._generate_key("test text", {"param": "value"})
        assert key4 == key5
    
    def test_clear(self):
        """Test svuotamento cache"""
        cache = CacheManager()
        
        # Aggiungi elementi
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.hits = 5
        cache.misses = 3
        
        # Svuota
        cache.clear()
        
        assert len(cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_get_stats(self):
        """Test statistiche cache"""
        cache = CacheManager(max_size=10)
        
        # Operazioni miste
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        cache.get("key3")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["size"] == 1
        assert stats["max_size"] == 10
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 1/3
        assert stats["enabled"] == True
    
    def test_contains(self):
        """Test operatore in"""
        cache = CacheManager()
        
        cache.set("key1", "value1")
        
        assert "key1" in cache
        assert "key2" not in cache 