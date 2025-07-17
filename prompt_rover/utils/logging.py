"""
Utilità di logging per Prompt Rover
"""

import logging
import sys
from pathlib import Path
from ..config import LOG_FORMAT, LOG_LEVEL, LOG_FILE

# Logger già configurati
_loggers = {}

def setup_logging():
    """
    Configura il sistema di logging
    """
    # Crea il formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Handler per console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Handler per file
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Configura il root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Logger specifici
    get_logger('performance')
    get_logger('visualization')
    get_logger('chat_mode')
    get_logger('concept_extraction')
    get_logger('embeddings')
    get_logger('graph')
    get_logger('cache')

def get_logger(name):
    """
    Ottiene un logger con il nome specificato
    
    Args:
        name: Nome del logger
        
    Returns:
        Logger configurato
    """
    if name not in _loggers:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
        _loggers[name] = logger
    
    return _loggers[name]

# Alias per compatibilità
perf_logger = get_logger('performance')
viz_logger = get_logger('visualization')
chat_logger = get_logger('chat_mode') 