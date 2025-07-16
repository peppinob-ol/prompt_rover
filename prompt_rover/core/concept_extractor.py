"""
Modulo per l'estrazione dei concetti dai testi
"""

import json
import spacy
from typing import List, Dict, Optional
import os

from ..config import (
    CONTENT_TYPE, LABEL, CATEGORY, DESCRIPTION,
    MAX_CONCEPTS_PER_TEXT, CONCEPT_MIN_LENGTH,
    LLM_API_TIMEOUT, DEFAULT_LLM_MODEL
)
from ..utils import get_logger, log_execution_time

logger = get_logger('concept_extraction')

class ConceptExtractor:
    """
    Estrae concetti dai testi usando LLM o metodi alternativi
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Inizializza l'estrattore di concetti
        
        Args:
            openai_api_key: Chiave API OpenAI (opzionale)
        """
        self.llm_client = None
        self.nlp = None
        
        if openai_api_key:
            self._initialize_openai(openai_api_key)
            
    def _initialize_openai(self, api_key: str) -> bool:
        """
        Inizializza il client OpenAI
        
        Args:
            api_key: Chiave API
            
        Returns:
            True se l'inizializzazione ha successo
        """
        try:
            from openai import OpenAI
            os.environ["OPENAI_API_KEY"] = api_key
            self.llm_client = OpenAI()
            logger.info("Client OpenAI inizializzato con successo")
            return True
        except ImportError:
            logger.error("Libreria OpenAI non installata")
            return False
        except Exception as e:
            logger.error(f"Errore nell'inizializzazione del client OpenAI: {str(e)}")
            return False
    
    def _load_spacy_model(self):
        """Carica il modello spaCy appropriato"""
        if self.nlp is None:
            try:
                self.nlp = spacy.load("it_core_news_sm")
                logger.info("Modello spaCy italiano caricato")
            except:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Modello spaCy inglese caricato")
                except:
                    self.nlp = spacy.load("xx_ent_wiki_sm")
                    logger.info("Modello spaCy multilingua caricato")
        return self.nlp
    
    @log_execution_time
    def extract_concepts(self, text: str, is_user: bool = True, 
                        use_llm: bool = True, model: str = DEFAULT_LLM_MODEL) -> List[Dict]:
        """
        Estrae concetti da un testo
        
        Args:
            text: Testo da analizzare
            is_user: True se il testo è dell'utente, False se dell'assistente
            use_llm: Se usare LLM per l'estrazione
            model: Modello OpenAI da usare
            
        Returns:
            Lista di concetti estratti
        """
        if use_llm and self.llm_client:
            return self._extract_with_llm(text, is_user, model)
        else:
            return self._extract_alternative(text, is_user)
    
    def _extract_with_llm(self, text: str, is_user: bool, model: str) -> List[Dict]:
        """
        Estrae concetti usando un LLM
        
        Args:
            text: Testo da analizzare
            is_user: True se il testo è dell'utente
            model: Modello da usare
            
        Returns:
            Lista di concetti
        """
        content_type = "user" if is_user else "assistant"
        logger.info(f"Estrazione concetti via LLM per {content_type}")
        
        prompt = f"""
        Analizza il seguente testo ed estrai i concetti chiave.

        TESTO:
        {text}

        ISTRUZIONI:
        1. Identifica i 5-10 concetti più significativi nel testo
        2. Per ogni concetto, fornisci:
        - Un'etichetta breve e precisa (massimo 5 parole)
        - Una categoria (entità, processo, relazione, attributo, ecc.)
        - Una breve descrizione del concetto nel contesto

        Restituisci SOLO un array JSON nel seguente formato, senza spiegazioni aggiuntive:
        [
            {{
                "label": "etichetta concetto",
                "category": "categoria",
                "description": "breve descrizione"
            }},
            ...
        ]
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Sei un assistente specializzato nell'estrazione di concetti dai testi."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                timeout=LLM_API_TIMEOUT
            )

            # Parsing della risposta
            concepts_json = response.choices[0].message.content
            
            # Estrai solo la parte JSON dalla risposta
            if "```json" in concepts_json:
                concepts_json = concepts_json.split("```json")[1].split("```")[0].strip()
            elif "```" in concepts_json:
                concepts_json = concepts_json.split("```")[1].split("```")[0].strip()

            concepts = json.loads(concepts_json)

            # Aggiungi metadata sull'origine
            for concept in concepts:
                concept["source"] = content_type
                concept[CONTENT_TYPE] = content_type

            return concepts[:MAX_CONCEPTS_PER_TEXT]
            
        except json.JSONDecodeError as e:
            logger.error(f"Errore nel parsing JSON: {e}")
            return self._extract_alternative(text, is_user)
        except TimeoutError as e:
            logger.error(f"Timeout nell'API OpenAI: {e}")
            return self._extract_alternative(text, is_user)
        except Exception as e:
            logger.error(f"Errore nella chiamata OpenAI: {str(e)}")
            return self._extract_alternative(text, is_user)
    
    @log_execution_time
    def _extract_alternative(self, text: str, is_user: bool) -> List[Dict]:
        """
        Metodo alternativo per estrarre concetti senza LLM
        
        Args:
            text: Testo da analizzare
            is_user: True se il testo è dell'utente
            
        Returns:
            Lista di concetti
        """
        content_type = "user" if is_user else "assistant"
        logger.info(f"Estrazione concetti via spaCy per {content_type}")
        
        # Carica il modello spaCy
        nlp = self._load_spacy_model()
        doc = nlp(text)
        
        concepts = []

        # Estrai entità nominate
        for ent in doc.ents:
            concepts.append({
                LABEL: ent.text,
                CATEGORY: ent.label_,
                DESCRIPTION: f"Entità di tipo {ent.label_}",
                CONTENT_TYPE: content_type
            })

        # Estrai chunk nominali significativi
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:  # Solo chunk con almeno 2 parole
                concepts.append({
                    LABEL: chunk.text,
                    CATEGORY: "noun_chunk",
                    DESCRIPTION: "Gruppo nominale significativo",
                    CONTENT_TYPE: content_type
                })

        # Deduplicazione basata sull'etichetta
        unique_concepts = []
        seen_labels = set()

        for concept in concepts:
            label = concept[LABEL].lower()
            if label not in seen_labels and len(label) > CONCEPT_MIN_LENGTH:
                seen_labels.add(label)
                unique_concepts.append(concept)

        # Se non abbiamo trovato abbastanza concetti, prendi anche parole singole importanti
        if len(unique_concepts) < 5:
            important_tokens = [
                token for token in doc 
                if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] 
                and len(token.text) > CONCEPT_MIN_LENGTH
            ]
            
            for token in important_tokens:
                if token.text.lower() not in seen_labels and len(unique_concepts) < MAX_CONCEPTS_PER_TEXT:
                    unique_concepts.append({
                        LABEL: token.text,
                        CATEGORY: token.pos_,
                        DESCRIPTION: f"Parola importante ({token.pos_})",
                        CONTENT_TYPE: content_type
                    })
                    seen_labels.add(token.text.lower())

        return unique_concepts[:MAX_CONCEPTS_PER_TEXT] 