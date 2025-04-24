import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import spacy
import gradio as gr; print(gr.__version__)
import traceback
import time
import logging
from functools import wraps

# Tentativo di importare IPython (opzionale)
try:
    from IPython.display import HTML, display as ipython_display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# Configura il logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output in console
        logging.FileHandler('concept_transform_detailed.log')
    ]
)

# Crea logger specifici
perf_logger = logging.getLogger('performance')
viz_logger = logging.getLogger('visualization')
chat_logger = logging.getLogger('chat_mode')

# Decoratore per misurare il tempo di esecuzione delle funzioni
def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Registra il tempo di esecuzione
        perf_logger.info(f"{func.__name__} eseguita in {execution_time:.4f} secondi")

        # Se l'esecuzione è lenta (> 1 secondo), registra come avviso
        if execution_time > 1.0:
            perf_logger.warning(f"Potenziale bottleneck: {func.__name__} ha impiegato {execution_time:.4f} secondi")

        return result
    return wrapper

# Costanti per colori e nomi di colonne
# Schema di colori
TEAL = "#3bb7b6"    # Colore utente/input
ORANGE = "#fbad52"  # Colore assistant/output

# Nomi delle colonne standardizzati
CONTENT_TYPE = "content_type"  # Valori: "user", "assistant" 
MESSAGE_ID = "message_id"      # Ordinamento messaggi
LABEL = "label"                # Etichetta concetto
CATEGORY = "category"          # Categoria concetto
DESCRIPTION = "description"    # Descrizione concetto
X_COORD = "x"                  # Coordinata X per visualizzazione
Y_COORD = "y"                  # Coordinata Y per visualizzazione
ALPHA = "alpha"                # Valore di trasparenza
EMBEDDING = "embedding"        # Embedding del concetto

# CLASSE PRINCIPALE
class ConceptTransformationVisualizer:
    def __init__(self, embedding_model="paraphrase-multilingual-MiniLM-L12-v2", openai_api_key=None):
        """
        Inizializza il visualizzatore di trasformazioni concettuali

        Args:
            embedding_model: Modello SentenceTransformer da utilizzare
            openai_api_key: Chiave API per OpenAI (opzionale)
        """

        start_time = time.time()
        perf_logger.info("Inizializzazione ConceptTransformationVisualizer...")

        self.embedding_model = None

        # Carica il modello di embedding con logging
        perf_logger.info(f"Caricamento modello embedding: {embedding_model}")
        model_start = time.time()
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            model_end = time.time()
            perf_logger.info(f"Modello embedding caricato in {model_end - model_start:.4f} secondi")
        except Exception as e:
            perf_logger.error(f"Errore caricamento modello: {str(e)}")
            raise

        # Inizializza client OpenAI se la chiave è fornita
        self.llm_client = None
        if openai_api_key:
            self.initialize_openai_client(openai_api_key)

        # Inizializzazione attributi per gli embedding (sempre, non solo con openai_api_key)
        self.name_embeddings = {}
        self.description_embeddings = {}  # Utilizzato come self.desc_embeddings
        self.desc_embeddings = {}  # Alias per compatibilità

        # Memorizzazione per embedding e configurazioni
        self.concept_embeddings = {
            'names': {},
            'descriptions': {}
        }
        self.umap_reducer = None
        self.tsne_reducer = None
        self.concept_graph = None

        # Inizializza strutture dati per modalità chat
        self.initialize_chat_mode()

        # Cache estesa
        self.concept_cache = {}      # Cache per concetti già calcolati
        self.embedding_cache = {}    # Cache per embedding già calcolati
        self.reduction_cache = {}    # Cache per riduzioni dimensionali
        self.graph_cache = {}        # Cache per grafi già calcolati

    def initialize_openai_client(self, api_key=None):
        """
        Centralizza l'inizializzazione del client OpenAI
        
        Args:
            api_key: Chiave API da utilizzare (override se specificata)
        
        Returns:
            bool: True se l'inizializzazione è riuscita, False altrimenti
        """
        key_to_use = api_key or os.environ.get("OPENAI_API_KEY", "")
        
        if not key_to_use.strip():
            perf_logger.warning("Nessuna API key OpenAI fornita")
            self.llm_client = None
            return False
        
        try:
            from openai import OpenAI
            # Imposta la chiave nell'ambiente
            os.environ["OPENAI_API_KEY"] = key_to_use
            
            # Crea il client
            self.llm_client = OpenAI()
            perf_logger.info("Client OpenAI inizializzato con successo")
            return True
        except ImportError:
            perf_logger.error("Libreria OpenAI non installata")
            self.llm_client = None
            return False
        except Exception as e:
            perf_logger.error(f"Errore nell'inizializzazione del client OpenAI: {str(e)}")
            self.llm_client = None
            return False

    def initialize_chat_mode(self):
        """Inizializza le strutture dati per la modalità chat"""
        chat_logger.info("Inizializzazione modalità chat")
        self.chat_concepts = []  # Lista di tutti i concetti estratti dalla chat
        self.chat_history = []   # Storia dei messaggi
        self.message_counter = 0 # Contatore messaggi
        self.concept_cache = {}  # Cache per concetti già calcolati
        self.concept_graph = None # Resetta anche il grafo

        # Assicurati che anche gli embedding siano reinizializzati
        self.name_embeddings = {}
        self.desc_embeddings = {}
        self.concept_embeddings = {
            'names': {},
            'descriptions': {}
        }
        chat_logger.info("Modalità chat inizializzata")
        return "Modalità chat inizializzata"

    def _get_cache_key(self, text, params=None):
        """
        Genera una chiave di cache univoca in base al testo e ai parametri
        
        Args:
            text: Testo da cui generare la chiave
            params: Dizionario di parametri aggiuntivi
            
        Returns:
            Stringa chiave per la cache
        """
        import hashlib
        
        # Genera hash dal testo
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Se non ci sono parametri, usa solo l'hash del testo
        if not params:
            return text_hash
        
        # Altrimenti, incorpora i parametri nella chiave
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        param_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()[:8]
        
        return f"{text_hash}_{param_hash}"

    @log_execution_time
    def extract_concepts_with_llm(self, text, is_user, model="gpt-4", timeout=30):
        """
        Estrae concetti da un testo utilizzando un modello LLM con gestione errori migliorata

        Args:
            text: Testo da cui estrarre i concetti
            is_user: True se il testo proviene dall'utente, False se dall'assistant
            model: Modello OpenAI da utilizzare
            timeout: Timeout in secondi per la chiamata API

        Returns:
            Lista di concetti estratti con metadati
        """
        content_type = "user" if is_user else "assistant"
        perf_logger.info(f"Estrazione concetti via LLM per {content_type}")
        
        # Verifica client OpenAI
        if not self.llm_client:
            perf_logger.warning("OpenAI API key non configurata. Uso metodo alternativo.")
            return self.extract_concepts_alternative(text, source_label)

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

        # Blocco try-except esterno per gestire errori API
        try:
            # Imposta timeout per la chiamata API
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Sei un assistente specializzato nell'estrazione di concetti da testi."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                timeout=timeout  # Aggiunta del timeout
            )

            # Blocco try-except interno per gestire errori di parsing
            try:
                concepts_json = response.choices[0].message.content
                # Estrai solo la parte JSON dalla risposta
                if "```json" in concepts_json:
                    concepts_json = concepts_json.split("```json")[1].split("```")[0].strip()
                elif "```" in concepts_json:
                    concepts_json = concepts_json.split("```")[1].split("```")[0].strip()

                concepts = json.loads(concepts_json)

                # Aggiungi metadati sulla sorgente
                for concept in concepts:
                    concept["source"] = content_type
                    concept[CONTENT_TYPE] = content_type

                return concepts
                
            except json.JSONDecodeError as e:
                # Errore specifico di parsing JSON
                perf_logger.error(f"Errore parsing JSON: {e}. Risposta: {response.choices[0].message.content}")
                perf_logger.info("Fallback al metodo alternativo di estrazione")
                return self.extract_concepts_alternative(text, source_label)
                
            except Exception as e:
                # Altri errori nel processing della risposta
                perf_logger.error(f"Errore nell'elaborazione della risposta: {e}")
                perf_logger.info("Fallback al metodo alternativo di estrazione")
                return self.extract_concepts_alternative(text, source_label)
                
        except TimeoutError as e:
            # Gestione timeout
            perf_logger.error(f"Timeout nell'API OpenAI: {e}")
            perf_logger.info("Fallback al metodo alternativo di estrazione")
            return self.extract_concepts_alternative(text, source_label)
            
        except Exception as e:
            # Qualsiasi altro errore di rete o API
            perf_logger.error(f"Errore nella chiamata all'API OpenAI: {str(e)}")
            perf_logger.info("Fallback al metodo alternativo di estrazione")
            return self.extract_concepts_alternative(text, source_label)

    @log_execution_time
    def extract_concepts_alternative(self, text, is_user):
        """
        Metodo alternativo per estrarre concetti senza LLM, usando spaCy

        Args:
            text: Testo da cui estrarre i concetti
            is_user: True se il testo proviene dall'utente, False se dall'assistant

        Returns:
            Lista di concetti estratti con metadati
        """
        content_type = "user" if is_user else "assistant"
        perf_logger.info(f"Estrazione concetti via spaCy per {content_type}")

        # Carica il modello spaCy per l'italiano
        try:
            nlp = spacy.load("it_core_news_sm")
        except:
            # Fallback al modello inglese se italiano non disponibile
            try:
                nlp = spacy.load("en_core_web_sm")
            except:
                # Ultimo tentativo con modello piccolo
                nlp = spacy.load("xx_ent_wiki_sm")

        # Misura tempo di parsing spaCy
        doc_start = time.time()
        doc = nlp(text)
        doc_end = time.time()
        perf_logger.info(f"Parsing spaCy completato in {doc_end - doc_start:.4f} secondi")

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
            if label not in seen_labels and len(label) > 3:
                seen_labels.add(label)
                unique_concepts.append(concept)

        # Se non abbiamo trovato abbastanza concetti, prendi anche parole singole importanti
        if len(unique_concepts) < 5:
            important_tokens = [token for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and len(token.text) > 3]
            for token in important_tokens:
                if token.text.lower() not in seen_labels and len(unique_concepts) < 10:
                    unique_concepts.append({
                        LABEL: token.text,
                        CATEGORY: token.pos_,
                        DESCRIPTION: f"Parola importante ({token.pos_})",
                        CONTENT_TYPE: content_type
                    })
                    seen_labels.add(token.text.lower())

        return unique_concepts[:10]  # Limita a 10 concetti

    @log_execution_time
    def compute_embeddings(self, concepts, name_weight=1.0):
        """
        Calcola gli embedding per una lista di concetti

        Args:
            concepts: Lista di concetti con metadati
            name_weight: Peso da dare al nome vs. descrizione (0.0-1.0)

        Returns:
            La stessa lista con embedding aggiuntivi
        """
        perf_logger.info(f"Calcolo embeddings per {len(concepts)} concetti")
        if len(concepts) == 0:
            return concepts

        # Estrai le etichette e descrizioni dei concetti
        labels = [concept[LABEL] for concept in concepts]
        descriptions = [concept.get(DESCRIPTION, concept[LABEL]) for concept in concepts]

        # Calcola gli embedding sia per nomi che per descrizioni
        name_embeddings = self.embedding_model.encode(labels)
        desc_embeddings = self.embedding_model.encode(descriptions)

        # Calcola embedding pesati
        for i, concept in enumerate(concepts):
            # Memorizza gli embedding separati per riferimento futuro
            self.name_embeddings[concept[LABEL]] = name_embeddings[i]
            self.desc_embeddings[concept[LABEL]] = desc_embeddings[i]
            
            # Aggiorna anche le strutture self.concept_embeddings
            self.concept_embeddings['names'][concept[LABEL]] = name_embeddings[i]
            self.concept_embeddings['descriptions'][concept[LABEL]] = desc_embeddings[i]

            # Calcola l'embedding pesato
            weighted_emb = name_weight * name_embeddings[i] + (1 - name_weight) * desc_embeddings[i]
            # Normalizza
            weighted_emb = weighted_emb / np.linalg.norm(weighted_emb)

            # Aggiungi al concetto
            concept[EMBEDDING] = weighted_emb
        
        return concepts

    @log_execution_time
    def build_concept_graph(self, concepts, k=3, use_mst=True, show_io_links=False):
        """
        Costruisce un grafo dei concetti (k-NN o MST)

        Args:
            concepts: Lista di concetti con embedding
            k: Numero di vicini più prossimi da collegare (per k-NN)
            use_mst: Se True, crea un Minimum Spanning Tree invece di k-NN
            show_io_links: Se True, mostra collegamenti diretti tra concetti input e output

        Returns:
            Grafo NetworkX
        """
        perf_logger.info(f"Costruzione grafo concettuale con {'MST' if use_mst else 'k-NN'}")

        if len(concepts) <= 1:
            # Crea un grafo vuoto o con un solo nodo
            G = nx.Graph()
            if len(concepts) == 1:
                G.add_node(
                    concepts[0][LABEL],
                    category=concepts[0].get(CATEGORY, ""),
                    description=concepts[0].get(DESCRIPTION, ""),
                    source=concepts[0].get(CONTENT_TYPE, "")
                )
            self.concept_graph = G
            return G

        # Crea un grafo vuoto
        G = nx.Graph()

        # Verifica se siamo in modalità chat
        message_linking = False
        if any(MESSAGE_ID in concept for concept in concepts):
            message_linking = True
            chat_logger.info("Modalità chat rilevata nella costruzione del grafo")

        # Aggiungi nodi con attributi
        for concept in concepts:
            node_attrs = {
                CATEGORY: concept.get(CATEGORY, ""),
                DESCRIPTION: concept.get(DESCRIPTION, ""),
                CONTENT_TYPE: concept.get(CONTENT_TYPE, ""),
                EMBEDDING: concept[EMBEDDING]
            }

            # Aggiungi attributi specifici per la modalità chat
            if message_linking and MESSAGE_ID in concept:
                node_attrs[MESSAGE_ID] = concept[MESSAGE_ID]
                node_attrs["message_type"] = concept.get("message_type", "")

            G.add_node(concept[LABEL], **node_attrs)

        # Se richiesto il Minimum Spanning Tree
        if use_mst:
            # Crea grafo completo con distanze
            G_complete = nx.Graph()

            # Aggiungi nodi
            for concept in concepts:
                G_complete.add_node(concept[LABEL])

            # Aggiungi archi con pesi basati sulla distanza
            labels = [concept[LABEL] for concept in concepts]
            embeddings = np.array([concept[EMBEDDING] for concept in concepts])

            for i, label_i in enumerate(labels):
                for j, label_j in enumerate(labels):
                    if i < j:  # Evita duplicati e auto-collegamenti
                        # Calcola distanza euclidea come peso
                        dist = np.linalg.norm(embeddings[i] - embeddings[j])
                        G_complete.add_edge(label_i, label_j, weight=dist)

            # Crea MST
            mst = nx.minimum_spanning_tree(G_complete)

            # Copia archi del MST nel grafo principale
            for u, v, data in mst.edges(data=True):
                # Inverti il peso per avere similarità invece di distanza
                similarity = 1.0 / (data["weight"] + 0.1)  # Evita divisione per zero
                G.add_edge(u, v, weight=similarity)

        else:
            # Comportamento precedente: crea k-NN
            # Calcola similarità coseno tra tutti i concetti
            labels = [concept[LABEL] for concept in concepts]
            embeddings = np.array([concept[EMBEDDING] for concept in concepts])

            # Normalizza gli embedding per calcolare più facilmente la similarità del coseno
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms

            # Calcola matrice di similarità del coseno
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

            # Per ogni nodo, aggiungi archi ai k nodi più simili
            for i, label in enumerate(labels):
                # Ottieni gli indici dei k concetti più simili (escluso se stesso)
                similarities = similarity_matrix[i]
                similarities[i] = -1  # Escludi se stesso
                top_k_indices = np.argsort(similarities)[-k:]

                # Aggiungi archi
                for idx in top_k_indices:
                    if similarities[idx] > 0:  # Controlla che ci sia una similarità positiva
                        G.add_edge(
                            label,
                            labels[idx],
                            weight=float(similarities[idx])
                        )

        # Aggiungi collegamenti espliciti tra concetti di input e output solo se richiesto
        if show_io_links:
            input_concepts = [c for c in concepts if c.get(CONTENT_TYPE, "") == "input"]
            output_concepts = [c for c in concepts if c.get(CONTENT_TYPE, "") == "output"]

            for input_concept in input_concepts:
                input_emb = input_concept[EMBEDDING]
                input_label = input_concept[LABEL]

                for output_concept in output_concepts:
                    output_emb = output_concept[EMBEDDING]
                    output_label = output_concept[LABEL]

                    # Calcola similarità coseno
                    sim = np.dot(input_emb, output_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(output_emb))

                    # Aggiungi arco se la similarità è alta
                    if sim > 0.7:
                        G.add_edge(
                            input_label,
                            output_label,
                            weight=float(sim),
                            type="input-output"
                        )

        # Se siamo in modalità chat, aggiungi collegamenti tra concetti dello stesso messaggio
        if message_linking:
            chat_logger.info("Aggiunta collegamenti intra-messaggio")
            message_groups = {}
            for concept in concepts:
                msg_id = concept.get(MESSAGE_ID)
                if msg_id:
                    if msg_id not in message_groups:
                        message_groups[msg_id] = []
                    message_groups[msg_id].append(concept[LABEL])

            # Aggiungi connessioni intra-messaggio con peso alto
            for msg_id, labels in message_groups.items():
                for i, label1 in enumerate(labels):
                    for label2 in labels[i+1:]:
                        if label1 in G.nodes and label2 in G.nodes:
                            G.add_edge(label1, label2, weight=0.9, type="same_message")

            chat_logger.info(f"Aggiunti collegamenti per {len(message_groups)} messaggi")

        self.concept_graph = G
        return G

    @log_execution_time
    def reduce_dimensions(self, concepts, method="umap", n_neighbors=5, min_dist=0.1, perplexity=30):
        """
        Riduce la dimensionalità degli embedding dei concetti con caching dei reducer

        Args:
            concepts: Lista di concetti con embedding
            method: Metodo di riduzione ('umap', 'tsne', o 'pca')
            n_neighbors: Parametro n_neighbors per UMAP
            min_dist: Parametro min_dist per UMAP
            perplexity: Parametro perplexity per t-SNE

        Returns:
            DataFrame con concetti e coordinate 2D
        """
        perf_logger.info(f"Riduzione dimensionalità con {method}")
        if len(concepts) == 0:
            return pd.DataFrame(columns=["label", "category", "description", "source", "x", "y"])
        
        # Estrai embeddings
        embeddings = np.array([concept["embedding"] for concept in concepts])
        
        # Gestisci il caso di un solo concetto
        if len(concepts) == 1:
            reduced_coords = np.array([[0.0, 0.0]])
        
        # Applica il metodo di riduzione scelto con caching
        elif method.lower() == "umap" and len(concepts) >= 4:
            # Usa reducer esistente se compatibile o crea nuovo
            if self.umap_reducer is not None and hasattr(self.umap_reducer, 'n_neighbors') and self.umap_reducer.n_neighbors == min(n_neighbors, len(concepts)-1):
                perf_logger.info("Riutilizzo reducer UMAP esistente")
                # Usa .transform() se dati sono compatibili con il training originale
                try:
                    reduced_coords = self.umap_reducer.transform(embeddings)
                except Exception:
                    # Altrimenti riadatta
                    reduced_coords = self.umap_reducer.fit_transform(embeddings)
            else:
                perf_logger.info("Creazione nuovo reducer UMAP")
                self.umap_reducer = UMAP(
                    n_neighbors=min(n_neighbors, len(concepts)-1),
                    min_dist=min_dist,
                    n_components=2,
                    metric='cosine',
                    random_state=42
                )
                reduced_coords = self.umap_reducer.fit_transform(embeddings)
        
        elif method.lower() == "tsne" and len(concepts) >= 3:
            # t-SNE non supporta transform(), quindi dobbiamo riadattare
            adjusted_perplexity = min(perplexity, len(concepts) // 3)
            adjusted_perplexity = max(5, adjusted_perplexity)  # Almeno 5
            
            self.tsne_reducer = TSNE(
                n_components=2,
                perplexity=adjusted_perplexity,
                random_state=42,
                init='pca',
                learning_rate='auto'
            )
            reduced_coords = self.tsne_reducer.fit_transform(embeddings)
        
        else:  # Default a PCA o per dataset piccoli
            # PCA supporta transform()
            if hasattr(self, 'pca_reducer') and self.pca_reducer is not None:
                try:
                    reduced_coords = self.pca_reducer.transform(embeddings)
                except:
                    self.pca_reducer = PCA(n_components=min(2, len(concepts))).fit(embeddings)
                    reduced_coords = self.pca_reducer.transform(embeddings)
            else:
                self.pca_reducer = PCA(n_components=min(2, len(concepts))).fit(embeddings)
                reduced_coords = self.pca_reducer.transform(embeddings)
                
            # Se abbiamo solo 2 concetti, allunghiamo artificialmente per renderli visibili
            if len(concepts) == 2:
                reduced_coords = reduced_coords * 2

        # Crea DataFrame
        df = pd.DataFrame({
            "label": [concept[LABEL] for concept in concepts],
            "category": [concept.get(CATEGORY, "") for concept in concepts],
            "description": [concept.get(DESCRIPTION, "") for concept in concepts],
            "source": [concept.get(CONTENT_TYPE, "") for concept in concepts],
            "x": reduced_coords[:, 0],
            "y": reduced_coords[:, 1]
        })

        # Aggiungi colonne specifiche per la modalità chat
        if any(MESSAGE_ID in concept for concept in concepts):
            df["message_id"] = [concept.get(MESSAGE_ID, 0) for concept in concepts]
            df["message_type"] = [concept.get("message_type", "") for concept in concepts]

        return df

    @log_execution_time
    # La vecchia funzione apply_umap ora chiama internamente reduce_dimensions
    def apply_umap(self, concepts, n_neighbors=5, min_dist=0.1):
        """
        Funzione legacy per compatibilità
        """
        return self.reduce_dimensions(
            concepts, method="umap",
            n_neighbors=n_neighbors, min_dist=min_dist
        )

    @log_execution_time
    def visualize_concepts(self, df, show_evolution=False, title=None):
        """
        Funzione di visualizzazione unificata per concetti
        
        Args:
            df: DataFrame con concetti e coordinate
            show_evolution: Se True, mostra evoluzione dei concetti con trasparenza
                        basata sull'età del messaggio
            title: Titolo personalizzato (opzionale)
        
        Returns:
            Figura matplotlib
        """
        viz_logger.info(f"Creazione visualizzazione concetti (evoluzione={show_evolution})")
        
        if df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Nessun concetto da visualizzare",
                ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig
            
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Mappa dei colori semplificata
        colors = {
            "user": TEAL,
            "assistant": ORANGE,
        }
        
        # Determina se siamo in modalità chat
        is_chat_mode = MESSAGE_ID in df.columns or "message_id" in df.columns
        
        # Standardizza i nomi delle colonne per compatibilità con codice esistente
        message_id_col = MESSAGE_ID if MESSAGE_ID in df.columns else "message_id"
        message_type_col = CONTENT_TYPE if CONTENT_TYPE in df.columns else "message_type"
        
        # Calcola alpha (trasparenza) basato sull'età del messaggio se richiesto
        if show_evolution and message_id_col in df.columns:
            max_message_id = df[message_id_col].max()
            df[ALPHA] = df[message_id_col].apply(
                lambda mid: max(0.3, 1.0 - (max_message_id - mid) / max(1, max_message_id))
            )
        else:
            df[ALPHA] = 0.7  # Alpha predefinito
        
        # Plot dei punti
        # Semplifica la logica di colorazione
        if is_chat_mode and message_type_col in df.columns:
            # Modalità chat - colora per tipo di messaggio
            for _, row in df.iterrows():
                alpha_value = row.get(ALPHA, 0.7)
                message_type = row.get(message_type_col, "")
                ax.scatter(
                    row[X_COORD] if X_COORD in df.columns else row["x"],
                    row[Y_COORD] if Y_COORD in df.columns else row["y"],
                    color=colors.get(message_type, "gray"),
                    alpha=alpha_value,
                    s=100,
                    edgecolors='black',
                    linewidth=0.5
                )
        else:
            # Modalità standard - colora per source (user/assistant)
            source_col = "source"  # Non cambiamo questo per retrocompatibilità
            
            # Mappa i valori nella colonna "source" ai colori corretti
            for source_value, group in df.groupby(source_col):
                # Semplifica la decisione sul colore
                color = colors.get(source_value, "gray")
                
                ax.scatter(
                    group[X_COORD] if X_COORD in df.columns else group["x"],
                    group[Y_COORD] if Y_COORD in df.columns else group["y"],
                    label=source_value,  # Usa direttamente source_value come label
                    color=color,
                    alpha=0.7,
                    s=100
                )
            ax.legend()
        
        # Aggiungi etichette ai nodi
        for _, row in df.iterrows():
            alpha_value = row.get(ALPHA, 0.7)
            x_coord = row[X_COORD] if X_COORD in df.columns else row["x"]
            y_coord = row[Y_COORD] if Y_COORD in df.columns else row["y"]
            label_text = row[LABEL] if LABEL in df.columns else row["label"]
            
            ax.annotate(
                label_text,
                (x_coord, y_coord),
                fontsize=9,
                alpha=alpha_value,
                xytext=(5, 5),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=alpha_value)
            )
        
        # Aggiungi archi del grafo se disponibile
        if self.concept_graph:
            # Crea il mapping delle posizioni dei nodi
            node_positions = {}
            for _, row in df.iterrows():
                node_label = row[LABEL] if LABEL in df.columns else row["label"]
                x_coord = row[X_COORD] if X_COORD in df.columns else row["x"]
                y_coord = row[Y_COORD] if Y_COORD in df.columns else row["y"]
                node_positions[node_label] = (x_coord, y_coord)
            
            # Calcola alpha per i nodi (per gli archi)
            node_alphas = {row[LABEL] if LABEL in df.columns else row["label"]: 
                        row.get(ALPHA, 0.7) for _, row in df.iterrows()}
            
            # Disegna tutti gli archi
            for u, v, data in self.concept_graph.edges(data=True):
                if u in node_positions and v in node_positions:
                    # Usa l'alpha minore tra i due nodi connessi
                    edge_alpha = min(node_alphas.get(u, 0.5), node_alphas.get(v, 0.5)) * 0.7
                    
                    # Stile diverso per tipi diversi di archi
                    if "type" in data:
                        if data["type"] == "same_message":
                            linestyle = '-'
                            color = 'purple'
                            linewidth = 1.0
                        elif data["type"] == "input-output":
                            linestyle = '--'
                            color = 'green'
                            linewidth = 1.2
                        else:
                            linestyle = '-'
                            color = 'black'
                            linewidth = 0.8
                    else:
                        linestyle = '-'
                        color = 'black'
                        linewidth = 0.8
                    
                    x = [node_positions[u][0], node_positions[v][0]]
                    y = [node_positions[u][1], node_positions[v][1]]
                    
                    ax.plot(x, y, linestyle=linestyle, color=color, alpha=edge_alpha, linewidth=linewidth)
        
        # Titolo appropriato per la modalità
        if title:
            ax.set_title(title, fontsize=14)
        elif show_evolution or is_chat_mode:
            ax.set_title('Evoluzione Concettuale della Conversazione', fontsize=14)
        else:
            ax.set_title('Mappa Concettuale della Trasformazione', fontsize=14)
        
        # Disattiva gli assi per un look più pulito
        ax.set_axis_off()
        plt.tight_layout()
        
        return fig

    @log_execution_time
    def visualize_static(self, df):
        """
        Wrapper per retrocompatibilità che chiama il metodo unificato
        """
        return self.visualize_concepts(df, show_evolution=False)

    @log_execution_time
    def visualize_static_chat(self, df):
        """
        Wrapper per retrocompatibilità che chiama il metodo unificato
        """
        return self.visualize_concepts(df, show_evolution=True)

    @log_execution_time
    def process_text_pair(self, input_text, output_text, use_llm=False,
                          name_weight=1.0, dim_reduction="umap",
                          use_mst=True, show_io_links=False):
        """
        Elabora una coppia di testi (input/output) e crea visualizzazioni, con caching

        Args:
            input_text: Testo di input
            output_text: Testo di output
            use_llm: Se usare LLM per l'estrazione dei concetti
            name_weight: Peso da dare al nome del concetto vs la descrizione (0.0-1.0)
            dim_reduction: Metodo di riduzione dimensionale ("umap", "tsne", o "pca")
            use_mst: Se usare Minimum Spanning Tree invece di k-NN
            show_io_links: Se mostrare collegamenti diretti input-output

        Returns:
            DataFrame e figure
        """
        perf_logger.info(f"Elaborazione coppia di testi - Lunghezza input: {len(input_text)} caratteri, output: {len(output_text)} caratteri")
        
        try:
            # Genera chiavi cache per input e output
            params = {
                "use_llm": use_llm,
                "name_weight": name_weight
            }
            input_cache_key = self._get_cache_key(input_text, params)
            output_cache_key = self._get_cache_key(output_text, params)
            
            # Estrai concetti (con cache)
            input_concepts = None
            if input_cache_key in self.concept_cache:
                perf_logger.info("Utilizzo concetti input dalla cache")
                input_concepts = self.concept_cache[input_cache_key]
            else:
                # Estrai concetti normalmente
                if use_llm and self.llm_client:
                    input_concepts = self.extract_concepts_with_llm(input_text, True)
                else:
                    input_concepts = self.extract_concepts_alternative(input_text, True)
                
                # Calcola embedding con peso nome/descrizione
                input_concepts = self.compute_embeddings(input_concepts, name_weight=name_weight)
                
                # Salva nella cache
                self.concept_cache[input_cache_key] = input_concepts
            
            # Ripeti per output
            output_concepts = None
            if output_cache_key in self.concept_cache:
                perf_logger.info("Utilizzo concetti output dalla cache")
                output_concepts = self.concept_cache[output_cache_key]
            else:
                # Estrai concetti normalmente
                if use_llm and self.llm_client:
                    output_concepts = self.extract_concepts_with_llm(output_text, False)
                else:
                    output_concepts = self.extract_concepts_alternative(output_text, False)
                
                # Calcola embedding con peso nome/descrizione
                output_concepts = self.compute_embeddings(output_concepts, name_weight=name_weight)
                
                # Salva nella cache
                self.concept_cache[output_cache_key] = output_concepts
            
            # Unisci concetti
            all_concepts = input_concepts + output_concepts
            
            # Genera chiave per il grafo
            graph_params = {
                "use_mst": use_mst,
                "show_io_links": show_io_links,
                "input_key": input_cache_key,
                "output_key": output_cache_key
            }
            graph_cache_key = self._get_cache_key("graph", graph_params)
            
            # Costruisci grafo con caching
            if graph_cache_key in self.graph_cache:
                perf_logger.info("Utilizzo grafo dalla cache")
                self.concept_graph = self.graph_cache[graph_cache_key]
            else:
                # Costruisci grafo normalmente
                self.concept_graph = self.build_concept_graph(all_concepts, use_mst=use_mst, show_io_links=show_io_links)
                # Salva nella cache
                self.graph_cache[graph_cache_key] = self.concept_graph
            
            # Genera chiave per la riduzione dimensionale
            reduction_params = {
                "method": dim_reduction,
                "input_key": input_cache_key,
                "output_key": output_cache_key
            }
            reduction_cache_key = self._get_cache_key("reduction", reduction_params)
            
            # Riduci dimensionalità con caching
            df = None
            if reduction_cache_key in self.reduction_cache:
                perf_logger.info(f"Utilizzo {dim_reduction} dalla cache")
                df = self.reduction_cache[reduction_cache_key]
            else:
                # Riduci dimensionalità normalmente
                df = self.reduce_dimensions(all_concepts, method=dim_reduction)
                # Salva nella cache
                self.reduction_cache[reduction_cache_key] = df
            
            # Crea visualizzazione statica
            fig_static = self.visualize_static_chat(df)
            
            return df, fig_static, "Elaborazione completata con successo (con caching ottimizzato)"
            
        except Exception as e:
            print(f"Errore nell'elaborazione: {e}")
            traceback.print_exc()
            # Crea dataframe e figure vuote in caso di errore
            empty_df = pd.DataFrame(columns=["label", "category", "description", "source", "x", "y"])
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Errore: {str(e)}", ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            
            return empty_df, fig, f"Errore: {str(e)}"

    @log_execution_time
    def process_new_message(self, message, is_user=True, name_weight=1.0,
                            dim_reduction_method="umap", use_mst=True,
                            show_io_links=False, use_llm=False):
        """
        Elabora un nuovo messaggio nella chat e aggiorna il grafo concettuale

        Args:
            message: Testo del messaggio
            is_user: True se il messaggio è dall'utente, False se è dal sistema
            name_weight: Peso da dare al nome del concetto vs la descrizione (0.0-1.0)
            dim_reduction_method: Metodo di riduzione dimensionale ("umap", "tsne", o "pca")
            use_mst: Se usare Minimum Spanning Tree invece di k-NN
            show_io_links: Se mostrare collegamenti diretti input-output
            use_llm: Se usare LLM per l'estrazione dei concetti

        Returns:
            DataFrame aggiornato e figure
        """
        content_type = "user" if is_user else "assistant"
        chat_logger.info(f"Elaborazione nuovo messaggio - {content_type} - Lunghezza: {len(message)} caratteri")

        try:
            # Incrementa contatore e salva il messaggio
            self.message_counter += 1
            message_id = self.message_counter

            self.chat_history.append({
                "id": message_id,
                "content": message,
                "is_user": is_user,
                "timestamp": time.time()
            })

            # Controlla se abbiamo già calcolato embedding per questo testo
            if message in self.concept_cache:
                chat_logger.info("Utilizzo concetti dalla cache")
                new_concepts = self.concept_cache[message]
            else:
                # Estrai concetti e calcola embedding
                chat_logger.info("Estrazione concetti dal messaggio")
                if use_llm and self.llm_client:
                    new_concepts = self.extract_concepts_with_llm(message, is_user=is_user)
                else:
                    new_concepts = self.extract_concepts_alternative(message, is_user=is_user)

                new_concepts = self.compute_embeddings(new_concepts, name_weight=name_weight)
                # Salva nella cache
                self.concept_cache[message] = new_concepts

            # Aggiungi metadati sul messaggio
            for concept in new_concepts:
                concept[MESSAGE_ID] = message_id
                concept["message_type"] = content_type

            # Aggiungi i nuovi concetti alla lista complessiva
            self.chat_concepts.extend(new_concepts)

            # Verifica limite di sicurezza sulla dimensione
            if len(self.chat_concepts) > 500:  # Limite conservativo
                return None, None, "Limite di concetti raggiunto, resetta la conversazione"

            # Ricostruisci il grafo con tutti i concetti
            self.build_concept_graph(self.chat_concepts, use_mst=use_mst, show_io_links=show_io_links)

            # Riduci dimensionalità con il metodo scelto
            df = self.reduce_dimensions(self.chat_concepts, method=dim_reduction_method)

            # Crea visualizzazione con la funzione unificata
            fig_static = self.visualize_concepts(df, show_evolution=True)

            return df, fig_static, f"Messaggio elaborato: trovati {len(new_concepts)} nuovi concetti"

        except Exception as e:
            chat_logger.error(f"Errore nell'elaborazione del messaggio: {e}", exc_info=True)
            # Crea dataframe e figure vuote in caso di errore
            empty_df = pd.DataFrame(columns=["label", "category", "description", "source", "x", "y"])
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Errore: {str(e)}", ha='center', va='center', fontsize=12)
            ax.set_axis_off()

            return empty_df, fig, f"Errore: {str(e)}"

    @log_execution_time
    def generate_chat_response(self, history):
        """
        Genera una risposta basata sulla conversazione usando OpenAI API
        senza system message personalizzato
        
        Args:
            history: Storia della conversazione in formato [{"role": "...", "content": "..."}]
            
        Returns:
            Stringa con la risposta generata dal modello
        """
        # Se non abbiamo un client OpenAI o non ci sono abbastanza messaggi
        if not self.llm_client or len(history) < 1:
            # Risposta fallback migliorata
            import random
            fallback_responses = [
                "Non riesco a comunicare con il modello oppure non hai fornito abbastanza messaggi.",
            ]
            return random.choice(fallback_responses)
    
        try:
            chat_logger.info("Generazione risposta tramite OpenAI")
            
            # Chiamata all'API con solo la conversazione esistente
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=history,
                temperature=0.7,
                max_tokens=150,
                timeout=10
            )
            
            # Estrai la risposta
            return response.choices[0].message.content
            
        except Exception as e:
            chat_logger.error(f"Errore nella generazione risposta: {str(e)}")
            return "Errore nella generazione risposta"


# Interfaccia GRADIO
def create_gradio_interface():
    visualizer = ConceptTransformationVisualizer()
    visualizer.initialize_chat_mode()  # Inizializza la chat all'avvio

    with gr.Blocks(title="Visualizzatore di Trasformazioni Concettuali") as demo:
        gr.Markdown("# Visualizzatore di Trasformazioni Concettuali")

        # Layout orizzontale principale
        with gr.Row():
            # Colonna sinistra per input con tabs
            with gr.Column(scale=1):
                with gr.Tabs() as tabs:
                    # Tab per modalità Input/Output
                    with gr.TabItem("Modalità Input/Output"):
                        input_text = gr.Textbox(label="Testo Input", placeholder="Inserisci il testo di input", lines=5)
                        output_text = gr.Textbox(label="Testo Output", placeholder="Inserisci il testo di output", lines=10)
                        with gr.Accordion("Opzioni avanzate", open=False):
                            name_weight_io = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.1,
                                                   label="Peso del nome rispetto alla descrizione")
                            dim_reduction_io = gr.Radio(["umap", "tsne", "pca"], label="Metodo di riduzione dimensionale", value="umap")
                            use_mst_io = gr.Checkbox(label="Usa Minimum Spanning Tree", value=True)
                            show_io_links_io = gr.Checkbox(label="Mostra collegamenti input-output", value=False)
                            use_llm_io = gr.Checkbox(label="Usa LLM per estrazione concetti", value=False)
                            openai_api_key_io = gr.Textbox(label="OpenAI API Key (opzionale)", placeholder="Inserisci se vuoi usare LLM")
                        submit_btn = gr.Button("Elabora e visualizza")
                        example_btn = gr.Button("Carica esempio fiori e saggezza")

                    # Tab per modalità Chat
                    with gr.TabItem("Modalità Chat"):
                        chatbot = gr.Chatbot(height=400, label="Conversazione", type="messages")  # Usa type="messages" per evitare warning
                        chat_msg = gr.Textbox(placeholder="Scrivi un messaggio qui...", label="Messaggio")
                        with gr.Accordion("Opzioni avanzate", open=False):
                            name_weight_chat = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.1,
                                                   label="Peso del nome rispetto alla descrizione")
                            dim_reduction_chat = gr.Radio(["umap", "tsne", "pca"], label="Metodo di riduzione dimensionale", value="umap")
                            use_mst_chat = gr.Checkbox(label="Usa Minimum Spanning Tree", value=True)
                            show_io_links_chat = gr.Checkbox(label="Mostra collegamenti input-output", value=False)
                            use_llm_chat = gr.Checkbox(label="Usa LLM per estrazione concetti", value=False)
                            openai_api_key_chat = gr.Textbox(label="OpenAI API Key (opzionale)", placeholder="Inserisci se vuoi usare LLM")
                        with gr.Row():
                            chat_submit = gr.Button("Invia")
                            chat_clear = gr.Button("Cancella conversazione")
                            reinit_btn = gr.Button("Reinizializza sistema")

            # Colonna destra per visualizzazione e risultati
            with gr.Column(scale=2):
                static_output = gr.Plot(label="Visualizzazione concetti")
                df_output = gr.DataFrame(label="Concetti estratti")
                status = gr.Textbox(label="Stato", interactive=False, value="Sistema inizializzato e pronto all'uso")

        # Funzione per reinizializzare la chat
        def reinit_chat():
            visualizer.initialize_chat_mode()
            return "Sistema reinizializzato"

        # Funzione per elaborare messaggi della chat
        def process_chat_message(message, history, name_weight, dim_reduction, use_mst, show_io_links, use_llm, openai_api_key):
            if not message.strip():
                return "", history, None, None, "Messaggio vuoto"

            # Configura OpenAI API Key se fornita e richiesta
            if use_llm and openai_api_key.strip():
                if not visualizer.initialize_openai_client(openai_api_key):
                    return "", history, None, None, "Errore nell'inizializzazione dell'API OpenAI"

            # Elabora il messaggio dell'utente
            try:
                if history is None:
                    history = []

                # Per il formato messages
                history.append({"role": "user", "content": message})

                # Analizza il messaggio dell'utente
                df, fig_static, status_msg = visualizer.process_new_message(
                    message, is_user=True,
                    name_weight=name_weight,
                    dim_reduction_method=dim_reduction,
                    use_mst=use_mst,
                    show_io_links=show_io_links,
                    use_llm=use_llm
                )

                #  risposta dell'assistant
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
                print(error_msg)
                traceback.print_exc()

                # Gestione dell'errore
                if history is None:
                    history = []
                history.append({"role": "assistant", "content": "Mi dispiace, si è verificato un errore nell'elaborazione"})
                return "", history, None, None, error_msg

        # Funzione per cancellare la chat
        def clear_chat():
            visualizer.initialize_chat_mode()
            return "", [], None, None, "Conversazione cancellata"

        # Carica esempio
        def load_example():
            input_text_example = "fiore saggezza"
            output_text_example = """
            Se pensiamo al simbolismo dei fiori in relazione alla saggezza, il fiore di loto è uno dei più rappresentativi. Nell'immaginario di molte culture, specialmente orientali, il loto cresce nel fango ma emerge puro e immacolato sulla superficie dell'acqua, simboleggiando la capacità di elevazione spirituale e intellettuale, proprio come chi acquisisce saggezza attraverso le difficoltà della vita.
            Anche la salvia ha un nome che richiama la saggezza, derivando dal latino salvus (salvo, sano). Questa pianta, oltre alle sue proprietà curative, è spesso associata alla prudenza e all'esperienza.
            Se pensiamo invece alla tradizione occidentale, il quercus robur, cioè la quercia, pur essendo un albero e non un fiore, è un simbolo di saggezza per la sua longevità e la sua capacità di resistere alle tempeste, proprio come una mente saggia che sa affrontare le avversità senza spezzarsi.
            """
            return input_text_example, output_text_example

        # Collegamenti per modalità Input/Output
        submit_btn.click(
            visualizer.process_text_pair,
            inputs=[input_text, output_text, use_llm_io, name_weight_io, dim_reduction_io, use_mst_io, show_io_links_io],
            outputs=[df_output, static_output, status]
        )
        example_btn.click(load_example, outputs=[input_text, output_text])

        # Collegamenti per modalità Chat
        chat_msg.submit(
            process_chat_message,
            inputs=[chat_msg, chatbot, name_weight_chat, dim_reduction_chat,
                    use_mst_chat, show_io_links_chat, use_llm_chat, openai_api_key_chat],
            outputs=[chat_msg, chatbot, df_output, static_output, status]
        )
        chat_submit.click(
            process_chat_message,
            inputs=[chat_msg, chatbot, name_weight_chat, dim_reduction_chat,
                    use_mst_chat, show_io_links_chat, use_llm_chat, openai_api_key_chat],
            outputs=[chat_msg, chatbot, df_output, static_output, status]
        )
        chat_clear.click(
            clear_chat,
            outputs=[chat_msg, chatbot, df_output, static_output, status]
        )

        # Collega il pulsante di reinizializzazione
        reinit_btn.click(
            reinit_chat,
            outputs=[status]
        )

    return demo

# Funzione per eseguire in Colab
def run_in_colab():
    """
    Esegue il visualizzatore in ambiente Google Colab
    """
    if IPYTHON_AVAILABLE:
        from IPython.display import display, HTML

        display(HTML(
            "<div style='background:#DDFFDD;border:1px solid #00CC00;padding:10px'>"
            "Interfaccia avviata! Se non viene visualizzata automaticamente, controlla che non sia stata bloccata dal browser."
            "</div>"
        ))

        # Crea e avvia l'interfaccia Gradio
        demo = create_gradio_interface()
        return demo.launch(inline=True, share=True, debug=True)
    else:
        print("IPython non disponibile. Esecuzione in ambiente locale")
        demo = create_gradio_interface()
        demo.launch(share=False, debug=True)

# Script principale
if __name__ == "__main__":
    try:
        # Verifica se siamo in ambiente Colab
        import google.colab
        # Se non genera eccezione, siamo in Colab
        print("Esecuzione in ambiente Google Colab")
        run_in_colab()
    except:
        # Se genera eccezione, siamo in ambiente locale
        print("Esecuzione in ambiente locale")
        demo = create_gradio_interface()
        demo.launch(share=False, debug=True)