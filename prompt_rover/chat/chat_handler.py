"""
Gestione della modalità chat per Prompt Rover
"""

import time
from typing import List, Dict, Optional, Tuple
import random

from ..config import MESSAGE_ID, MAX_CHAT_CONCEPTS
from ..utils import get_logger

logger = get_logger('chat_mode')

class ChatHandler:
    """
    Gestisce la modalità chat, mantenendo lo stato della conversazione
    """
    
    def __init__(self, llm_client=None):
        """
        Inizializza il gestore della chat
        
        Args:
            llm_client: Client OpenAI opzionale per generare risposte
        """
        self.llm_client = llm_client
        self.reset()
    
    def reset(self):
        """Resetta lo stato della chat"""
        logger.info("Inizializzazione modalità chat")
        self.chat_concepts = []      # Lista di tutti i concetti estratti
        self.chat_history = []       # Storico messaggi
        self.message_counter = 0     # Contatore messaggi
        logger.info("Modalità chat inizializzata")
    
    def add_message(self, content: str, is_user: bool = True) -> Dict:
        """
        Aggiunge un messaggio alla conversazione
        
        Args:
            content: Contenuto del messaggio
            is_user: True se messaggio dell'utente, False se assistente
            
        Returns:
            Dizionario con info del messaggio
        """
        self.message_counter += 1
        
        message = {
            "id": self.message_counter,
            "content": content,
            "is_user": is_user,
            "role": "user" if is_user else "assistant",
            "timestamp": time.time()
        }
        
        self.chat_history.append(message)
        logger.info(f"Aggiunto messaggio #{self.message_counter} dalla {'utente' if is_user else 'assistente'}")
        
        return message
    
    def update_concepts(self, new_concepts: List[Dict], message_id: int, 
                       content_type: str = "user"):
        """
        Aggiorna i concetti con nuovi concetti da un messaggio
        
        Args:
            new_concepts: Nuovi concetti estratti
            message_id: ID del messaggio
            content_type: Tipo di contenuto ("user" o "assistant")
            
        Returns:
            Numero totale di concetti
        """
        # Aggiungi metadata del messaggio
        for concept in new_concepts:
            concept[MESSAGE_ID] = message_id
            concept["message_type"] = content_type
        
        # Aggiungi alla lista complessiva
        self.chat_concepts.extend(new_concepts)
        
        logger.info(f"Aggiunti {len(new_concepts)} concetti dal messaggio #{message_id}")
        
        # Verifica limite di sicurezza
        if len(self.chat_concepts) > MAX_CHAT_CONCEPTS:
            logger.warning(f"Raggiunto limite concetti: {len(self.chat_concepts)}")
        
        return len(self.chat_concepts)
    
    def check_concept_limit(self) -> bool:
        """
        Verifica se abbiamo raggiunto il limite di concetti
        
        Returns:
            True se il limite è stato raggiunto
        """
        return len(self.chat_concepts) > MAX_CHAT_CONCEPTS
    
    def generate_response(self, model: str = "gpt-3.5-turbo",
                         temperature: float = 0.7,
                         max_tokens: int = 150) -> str:
        """
        Genera una risposta basata sulla conversazione
        
        Args:
            model: Modello da usare
            temperature: Temperatura per la generazione
            max_tokens: Numero massimo di token
            
        Returns:
            Risposta generata
        """
        if not self.llm_client or len(self.chat_history) < 1:
            return self._get_fallback_response()
        
        try:
            logger.info("Generazione risposta via OpenAI")
            
            # Prepara messaggi per l'API
            messages = []
            for msg in self.chat_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Chiamata API
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=10
            )
            
            # Estrai risposta
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Errore nella generazione risposta: {str(e)}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """Genera una risposta di fallback"""
        fallback_responses = [
            "Non posso comunicare con il modello o non hai fornito abbastanza messaggi.",
            "Mi dispiace, c'è stato un problema nella generazione della risposta.",
            "Interessante! Dimmi di più...",
            "Capisco. Vuoi approfondire questo punto?"
        ]
        return random.choice(fallback_responses)
    
    def get_conversation_stats(self) -> Dict:
        """
        Ottiene statistiche sulla conversazione
        
        Returns:
            Dizionario con statistiche
        """
        user_messages = sum(1 for msg in self.chat_history if msg["is_user"])
        assistant_messages = len(self.chat_history) - user_messages
        
        return {
            "total_messages": len(self.chat_history),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "total_concepts": len(self.chat_concepts),
            "messages_with_concepts": len(set(c.get(MESSAGE_ID, 0) for c in self.chat_concepts))
        }
    
    def get_message_concepts(self, message_id: int) -> List[Dict]:
        """
        Ottiene i concetti di un messaggio specifico
        
        Args:
            message_id: ID del messaggio
            
        Returns:
            Lista di concetti del messaggio
        """
        return [c for c in self.chat_concepts if c.get(MESSAGE_ID) == message_id]
    
    def get_conversation_summary(self) -> str:
        """
        Genera un riassunto della conversazione
        
        Returns:
            Stringa con il riassunto
        """
        stats = self.get_conversation_stats()
        
        summary = f"Conversazione con {stats['total_messages']} messaggi:\n"
        summary += f"- Messaggi utente: {stats['user_messages']}\n"
        summary += f"- Messaggi assistente: {stats['assistant_messages']}\n"
        summary += f"- Concetti estratti: {stats['total_concepts']}\n"
        
        # Trova i concetti più frequenti
        concept_labels = [c.get("label", "") for c in self.chat_concepts]
        from collections import Counter
        most_common = Counter(concept_labels).most_common(5)
        
        if most_common:
            summary += "\nConcetti più frequenti:\n"
            for label, count in most_common:
                summary += f"- {label}: {count} volte\n"
        
        return summary 