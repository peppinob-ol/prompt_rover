#main.py

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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

# Support for .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load variables from .env
    DOTENV_LOADED = True
except ImportError:
    DOTENV_LOADED = False
    perf_logger.warning("python-dotenv not installed. Install with 'pip install python-dotenv' for .env file support")

# Attempt to import IPython (optional)
try:
    from IPython.display import HTML, display as ipython_display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('concept_transform_detailed.log')
    ]
)

# Create specific loggers
perf_logger = logging.getLogger('performance')
viz_logger = logging.getLogger('visualization')
chat_logger = logging.getLogger('chat_mode')

# Decorator to measure execution time of functions
def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Log the execution time
        perf_logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")

        # If execution is slow (> 1 second), log as warning
        if execution_time > 1.0:
            perf_logger.warning(f"Potential bottleneck: {func.__name__} took {execution_time:.4f} seconds")

        return result
    return wrapper

# Constants for colors and column names
# Color schema
TEAL = "#3bb7b6"    # User/input color
ORANGE = "#fbad52"  # Assistant/output color

# Standardized column names
CONTENT_TYPE = "content_type"  # Values: "user", "assistant" 
MESSAGE_ID = "message_id"      # Message ordering
LABEL = "label"                # Concept label
CATEGORY = "category"          # Concept category
DESCRIPTION = "description"    # Concept description
X_COORD = "x"                  # X coordinate for visualization
Y_COORD = "y"                  # Y coordinate for visualization
ALPHA = "alpha"                # Transparency value
EMBEDDING = "embedding"        # Concept embedding

# MAIN CLASS
class ConceptTransformationVisualizer:
    def __init__(self, embedding_model="paraphrase-multilingual-MiniLM-L12-v2", openai_api_key=None):
        """
        Initializes the concept transformation visualizer

        Args:
            embedding_model: SentenceTransformer model to use
            openai_api_key: API key for OpenAI (optional)
        """

        start_time = time.time()
        perf_logger.info("Initializing ConceptTransformationVisualizer...")

        self.embedding_model = None

        # Load the embedding model with logging
        perf_logger.info(f"Loading embedding model: {embedding_model}")
        model_start = time.time()
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            model_end = time.time()
            perf_logger.info(f"Embedding model loaded in {model_end - model_start:.4f} seconds")
        except Exception as e:
            perf_logger.error(f"Error loading model: {str(e)}")
            raise

        # Initialize OpenAI client if key is provided
        self.llm_client = None
        if openai_api_key:
            self.initialize_openai_client(openai_api_key)

        # Initialize attributes for embeddings (always, not just with openai_api_key)
        self.name_embeddings = {}
        self.description_embeddings = {}  # Used as self.desc_embeddings
        self.desc_embeddings = {}  # Alias for compatibility

        # Storage for embeddings and configurations
        self.concept_embeddings = {
            'names': {},
            'descriptions': {}
        }
        self.umap_reducer = None
        self.tsne_reducer = None
        self.concept_graph = None

        # Initialize data structures for chat mode
        self.initialize_chat_mode()

        # Extended cache
        self.concept_cache = {}      # Cache for already calculated concepts
        self.embedding_cache = {}    # Cache for already calculated embeddings
        self.reduction_cache = {}    # Cache for dimensional reductions
        self.graph_cache = {}        # Cache for already calculated graphs

    def initialize_openai_client(self, api_key=None):
        """
        Centralizes the initialization of the OpenAI client
        
        Args:
            api_key: API key to use (override if specified)
        
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        key_to_use = api_key or os.environ.get("OPENAI_API_KEY", "")
        
        if not key_to_use.strip():
            perf_logger.warning("No OpenAI API key provided")
            self.llm_client = None
            return False
        
        try:
            from openai import OpenAI
            # Set the key in the environment
            os.environ["OPENAI_API_KEY"] = key_to_use
            
            # Create the client
            self.llm_client = OpenAI()
            perf_logger.info("OpenAI client initialized successfully")
            return True
        except ImportError:
            perf_logger.error("OpenAI library not installed")
            self.llm_client = None
            return False
        except Exception as e:
            perf_logger.error(f"Error initializing OpenAI client: {str(e)}")
            self.llm_client = None
            return False

    def initialize_chat_mode(self):
        """Initializes data structures for chat mode"""
        chat_logger.info("Initializing chat mode")
        self.chat_concepts = []  # List of all concepts extracted from chat
        self.chat_history = []   # Message history
        self.message_counter = 0 # Message counter
        self.concept_cache = {}  # Cache for already calculated concepts
        self.concept_graph = None # Also reset the graph

        # Make sure embeddings are also reinitialized
        self.name_embeddings = {}
        self.desc_embeddings = {}
        self.concept_embeddings = {
            'names': {},
            'descriptions': {}
        }
        chat_logger.info("Chat mode initialized")
        return "Chat mode initialized"

    def _get_cache_key(self, text, params=None):
        """
        Generates a unique cache key based on text and parameters
        
        Args:
            text: Text from which to generate the key
            params: Dictionary of additional parameters
            
        Returns:
            String key for the cache
        """
        import hashlib
        
        # Generate hash from text
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # If there are no parameters, use just the text hash
        if not params:
            return text_hash
        
        # Otherwise, incorporate parameters in the key
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        param_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()[:8]
        
        return f"{text_hash}_{param_hash}"

    @log_execution_time
    def extract_concepts_with_llm(self, text, is_user, model="gpt-4", timeout=30):
        """
        Extracts concepts from a text using an LLM model with improved error handling

        Args:
            text: Text from which to extract concepts
            is_user: True if the text is from the user, False if from the assistant
            model: OpenAI model to use
            timeout: Timeout in seconds for the API call

        Returns:
            List of extracted concepts with metadata
        """
        content_type = "user" if is_user else "assistant"
        perf_logger.info(f"Extracting concepts via LLM for {content_type}")
        
        # Verify OpenAI client
        if not self.llm_client:
            perf_logger.warning("OpenAI API key not configured. Using alternative method.")
            return self.extract_concepts_alternative(text, is_user)

        prompt = f"""
        Analyze the following text and extract key concepts.

        TEXT:
        {text}

        INSTRUCTIONS:
        1. Identify the 5-10 most significant concepts in the text
        2. For each concept, provide:
        - A short and precise label (maximum 5 words)
        - A category (entity, process, relationship, attribute, etc.)
        - A brief description of the concept in context

        Return ONLY a JSON array in the following format, without additional explanations:
        [
            {{
                "label": "concept label",
                "category": "category",
                "description": "brief description"
            }},
            ...
        ]
        """

        # Outer try-except block to handle API errors
        try:
            # Set timeout for API call
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an assistant specialized in extracting concepts from texts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                timeout=timeout  # Add timeout
            )

            # Inner try-except block to handle parsing errors
            try:
                concepts_json = response.choices[0].message.content
                # Extract only the JSON part from the response
                if "```json" in concepts_json:
                    concepts_json = concepts_json.split("```json")[1].split("```")[0].strip()
                elif "```" in concepts_json:
                    concepts_json = concepts_json.split("```")[1].split("```")[0].strip()

                concepts = json.loads(concepts_json)

                # Add metadata about the source
                for concept in concepts:
                    concept["source"] = content_type
                    concept[CONTENT_TYPE] = content_type

                return concepts
                
            except json.JSONDecodeError as e:
                # Specific JSON parsing error
                perf_logger.error(f"JSON parsing error: {e}. Response: {response.choices[0].message.content}")
                perf_logger.info("Falling back to alternative extraction method")
                return self.extract_concepts_alternative(text, is_user)
                
            except Exception as e:
                # Other errors in processing the response
                perf_logger.error(f"Error processing the response: {e}")
                perf_logger.info("Falling back to alternative extraction method")
                return self.extract_concepts_alternative(text, is_user)
                
        except TimeoutError as e:
            # Timeout handling
            perf_logger.error(f"Timeout in OpenAI API: {e}")
            perf_logger.info("Falling back to alternative extraction method")
            return self.extract_concepts_alternative(text, is_user)
            
        except Exception as e:
            # Any other network or API error
            perf_logger.error(f"Error calling OpenAI API: {str(e)}")
            perf_logger.info("Falling back to alternative extraction method")
            return self.extract_concepts_alternative(text, is_user)

    @log_execution_time
    def extract_concepts_alternative(self, text, is_user):
        """
        Alternative method to extract concepts without LLM, using spaCy

        Args:
            text: Text from which to extract concepts
            is_user: True if the text is from the user, False if from the assistant

        Returns:
            List of extracted concepts with metadata
        """
        content_type = "user" if is_user else "assistant"
        perf_logger.info(f"Extracting concepts via spaCy for {content_type}")

        # Load spaCy model for the appropriate language
        try:
            nlp = spacy.load("it_core_news_sm")
        except:
            # Fallback to English model if Italian is not available
            try:
                nlp = spacy.load("en_core_web_sm")
            except:
                # Last attempt with small model
                nlp = spacy.load("xx_ent_wiki_sm")

        # Measure spaCy parsing time
        doc_start = time.time()
        doc = nlp(text)
        doc_end = time.time()
        perf_logger.info(f"spaCy parsing completed in {doc_end - doc_start:.4f} seconds")

        concepts = []

        # Extract named entities
        for ent in doc.ents:
            concepts.append({
                LABEL: ent.text,
                CATEGORY: ent.label_,
                DESCRIPTION: f"Entity of type {ent.label_}",
                CONTENT_TYPE: content_type
            })

        # Extract significant noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:  # Only chunks with at least 2 words
                concepts.append({
                    LABEL: chunk.text,
                    CATEGORY: "noun_chunk",
                    DESCRIPTION: "Significant nominal group",
                    CONTENT_TYPE: content_type
                })

        # Deduplication based on label
        unique_concepts = []
        seen_labels = set()

        for concept in concepts:
            label = concept[LABEL].lower()
            if label not in seen_labels and len(label) > 3:
                seen_labels.add(label)
                unique_concepts.append(concept)

        # If we haven't found enough concepts, also take single important words
        if len(unique_concepts) < 5:
            important_tokens = [token for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and len(token.text) > 3]
            for token in important_tokens:
                if token.text.lower() not in seen_labels and len(unique_concepts) < 10:
                    unique_concepts.append({
                        LABEL: token.text,
                        CATEGORY: token.pos_,
                        DESCRIPTION: f"Important word ({token.pos_})",
                        CONTENT_TYPE: content_type
                    })
                    seen_labels.add(token.text.lower())

        return unique_concepts[:10]  # Limit to 10 concepts

    @log_execution_time
    def compute_embeddings(self, concepts, name_weight=1.0):
        """
        Computes embeddings for a list of concepts

        Args:
            concepts: List of concepts with metadata
            name_weight: Weight to give to the name vs. description (0.0-1.0)

        Returns:
            The same list with additional embeddings
        """
        perf_logger.info(f"Computing embeddings for {len(concepts)} concepts")
        if len(concepts) == 0:
            return concepts

        # Extract concept labels and descriptions
        labels = [concept[LABEL] for concept in concepts]
        descriptions = [concept.get(DESCRIPTION, concept[LABEL]) for concept in concepts]

        # Compute embeddings for both names and descriptions
        name_embeddings = self.embedding_model.encode(labels)
        desc_embeddings = self.embedding_model.encode(descriptions)

        # Compute weighted embeddings
        for i, concept in enumerate(concepts):
            # Store separate embeddings for future reference
            self.name_embeddings[concept[LABEL]] = name_embeddings[i]
            self.desc_embeddings[concept[LABEL]] = desc_embeddings[i]
            
            # Also update self.concept_embeddings structures
            self.concept_embeddings['names'][concept[LABEL]] = name_embeddings[i]
            self.concept_embeddings['descriptions'][concept[LABEL]] = desc_embeddings[i]

            # Compute weighted embedding
            weighted_emb = name_weight * name_embeddings[i] + (1 - name_weight) * desc_embeddings[i]
            # Normalize
            weighted_emb = weighted_emb / np.linalg.norm(weighted_emb)

            # Add to concept
            concept[EMBEDDING] = weighted_emb
        
        return concepts

    @log_execution_time
    def build_concept_graph(self, concepts, k=3, use_mst=True, show_io_links=False):
        """
        Builds a concept graph (k-NN or MST)

        Args:
            concepts: List of concepts with embeddings
            k: Number of nearest neighbors to connect (for k-NN)
            use_mst: If True, creates a Minimum Spanning Tree instead of k-NN
            show_io_links: If True, shows direct connections between input and output concepts

        Returns:
            NetworkX graph
        """
        perf_logger.info(f"Building concept graph with {'MST' if use_mst else 'k-NN'}")

        if len(concepts) <= 1:
            # Create an empty graph or with just one node
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

        # Create an empty graph
        G = nx.Graph()

        # Check if we're in chat mode
        message_linking = False
        if any(MESSAGE_ID in concept for concept in concepts):
            message_linking = True
            chat_logger.info("Chat mode detected in graph construction")

        # Add nodes with attributes
        for concept in concepts:
            node_attrs = {
                CATEGORY: concept.get(CATEGORY, ""),
                DESCRIPTION: concept.get(DESCRIPTION, ""),
                CONTENT_TYPE: concept.get(CONTENT_TYPE, ""),
                EMBEDDING: concept[EMBEDDING]
            }

            # Add specific attributes for chat mode
            if message_linking and MESSAGE_ID in concept:
                node_attrs[MESSAGE_ID] = concept[MESSAGE_ID]
                node_attrs["message_type"] = concept.get("message_type", "")

            G.add_node(concept[LABEL], **node_attrs)

        # If Minimum Spanning Tree is requested
        if use_mst:
            # Create complete graph with distances
            G_complete = nx.Graph()

            # Add nodes
            for concept in concepts:
                G_complete.add_node(concept[LABEL])

            # Add edges with weights based on distance
            labels = [concept[LABEL] for concept in concepts]
            embeddings = np.array([concept[EMBEDDING] for concept in concepts])

            for i, label_i in enumerate(labels):
                for j, label_j in enumerate(labels):
                    if i < j:  # Avoid duplicates and self-connections
                        # Calculate Euclidean distance as weight
                        dist = np.linalg.norm(embeddings[i] - embeddings[j])
                        G_complete.add_edge(label_i, label_j, weight=dist)

            # Create MST
            mst = nx.minimum_spanning_tree(G_complete)

            # Copy MST edges to main graph
            for u, v, data in mst.edges(data=True):
                # Invert weight to have similarity instead of distance
                similarity = 1.0 / (data["weight"] + 0.1)  # Avoid division by zero
                G.add_edge(u, v, weight=similarity)

        else:
            # Previous behavior: create k-NN
            # Calculate cosine similarity between all concepts
            labels = [concept[LABEL] for concept in concepts]
            embeddings = np.array([concept[EMBEDDING] for concept in concepts])

            # Normalize embeddings to more easily compute cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms

            # Compute cosine similarity matrix
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

            # For each node, add edges to the k most similar nodes
            for i, label in enumerate(labels):
                # Get indices of k most similar concepts (excluding self)
                similarities = similarity_matrix[i]
                similarities[i] = -1  # Exclude self
                top_k_indices = np.argsort(similarities)[-k:]

                # Add edges
                for idx in top_k_indices:
                    if similarities[idx] > 0:  # Check that there's a positive similarity
                        G.add_edge(
                            label,
                            labels[idx],
                            weight=float(similarities[idx])
                        )

        # Add explicit connections between input and output concepts only if requested
        if show_io_links:
            input_concepts = [c for c in concepts if c.get(CONTENT_TYPE, "") == "input"]
            output_concepts = [c for c in concepts if c.get(CONTENT_TYPE, "") == "output"]

            for input_concept in input_concepts:
                input_emb = input_concept[EMBEDDING]
                input_label = input_concept[LABEL]

                for output_concept in output_concepts:
                    output_emb = output_concept[EMBEDDING]
                    output_label = output_concept[LABEL]

                    # Calculate cosine similarity
                    sim = np.dot(input_emb, output_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(output_emb))

                    # Add edge if similarity is high
                    if sim > 0.7:
                        G.add_edge(
                            input_label,
                            output_label,
                            weight=float(sim),
                            type="input-output"
                        )

        # If we're in chat mode, add connections between concepts from the same message
        if message_linking:
            chat_logger.info("Adding intra-message connections")
            message_groups = {}
            for concept in concepts:
                msg_id = concept.get(MESSAGE_ID)
                if msg_id:
                    if msg_id not in message_groups:
                        message_groups[msg_id] = []
                    message_groups[msg_id].append(concept[LABEL])

            # Add intra-message connections with high weight
            for msg_id, labels in message_groups.items():
                for i, label1 in enumerate(labels):
                    for label2 in labels[i+1:]:
                        if label1 in G.nodes and label2 in G.nodes:
                            G.add_edge(label1, label2, weight=0.9, type="same_message")

            chat_logger.info(f"Added connections for {len(message_groups)} messages")

        self.concept_graph = G
        return G

    @log_execution_time
    def reduce_dimensions(self, concepts, method="umap", n_neighbors=5, min_dist=0.1, perplexity=30):
        """
        Reduces the dimensionality of concept embeddings with reducer caching

        Args:
            concepts: List of concepts with embeddings
            method: Reduction method ('umap', 'tsne', or 'pca')
            n_neighbors: n_neighbors parameter for UMAP
            min_dist: min_dist parameter for UMAP
            perplexity: perplexity parameter for t-SNE

        Returns:
            DataFrame with concepts and 2D coordinates
        """
        perf_logger.info(f"Reducing dimensionality with {method}")
        if len(concepts) == 0:
            return pd.DataFrame(columns=["label", "category", "description", "source", "x", "y"])
        
        # Extract embeddings
        embeddings = np.array([concept["embedding"] for concept in concepts])
        
        # Handle the case of a single concept
        if len(concepts) == 1:
            reduced_coords = np.array([[0.0, 0.0]])
        
        # Apply the chosen reduction method with caching
        elif method.lower() == "umap" and len(concepts) >= 4:
            # Use existing reducer if compatible or create new one
            if self.umap_reducer is not None and hasattr(self.umap_reducer, 'n_neighbors') and self.umap_reducer.n_neighbors == min(n_neighbors, len(concepts)-1):
                perf_logger.info("Reusing existing UMAP reducer")
                # Use .transform() if data is compatible with original training
                try:
                    reduced_coords = self.umap_reducer.transform(embeddings)
                except Exception:
                    # Otherwise, refit
                    reduced_coords = self.umap_reducer.fit_transform(embeddings)
            else:
                perf_logger.info("Creating new UMAP reducer")
                self.umap_reducer = UMAP(
                    n_neighbors=min(n_neighbors, len(concepts)-1),
                    min_dist=min_dist,
                    n_components=2,
                    metric='cosine',
                    random_state=42
                )
                reduced_coords = self.umap_reducer.fit_transform(embeddings)
        
        elif method.lower() == "tsne" and len(concepts) >= 3:
            # t-SNE doesn't support transform(), so we need to refit
            adjusted_perplexity = min(perplexity, len(concepts) // 3)
            adjusted_perplexity = max(5, adjusted_perplexity)  # At least 5
            
            self.tsne_reducer = TSNE(
                n_components=2,
                perplexity=adjusted_perplexity,
                random_state=42,
                init='pca',
                learning_rate='auto'
            )
            reduced_coords = self.tsne_reducer.fit_transform(embeddings)
        
        else:  # Default to PCA or for small datasets
            # PCA supports transform()
            if hasattr(self, 'pca_reducer') and self.pca_reducer is not None:
                try:
                    reduced_coords = self.pca_reducer.transform(embeddings)
                except:
                    self.pca_reducer = PCA(n_components=min(2, len(concepts))).fit(embeddings)
                    reduced_coords = self.pca_reducer.transform(embeddings)
            else:
                self.pca_reducer = PCA(n_components=min(2, len(concepts))).fit(embeddings)
                reduced_coords = self.pca_reducer.transform(embeddings)
                
            # If we only have 2 concepts, artificially stretch them to make them visible
            if len(concepts) == 2:
                reduced_coords = reduced_coords * 2

        # Create DataFrame
        df = pd.DataFrame({
            "label": [concept[LABEL] for concept in concepts],
            "category": [concept.get(CATEGORY, "") for concept in concepts],
            "description": [concept.get(DESCRIPTION, "") for concept in concepts],
            "source": [concept.get(CONTENT_TYPE, "") for concept in concepts],
            "x": reduced_coords[:, 0],
            "y": reduced_coords[:, 1]
        })

        # Add specific columns for chat mode
        if any(MESSAGE_ID in concept for concept in concepts):
            df["message_id"] = [concept.get(MESSAGE_ID, 0) for concept in concepts]
            df["message_type"] = [concept.get("message_type", "") for concept in concepts]

        return df

    @log_execution_time
    # The old apply_umap function now internally calls reduce_dimensions
    def apply_umap(self, concepts, n_neighbors=5, min_dist=0.1):
        """
        Legacy function for compatibility
        """
        return self.reduce_dimensions(
            concepts, method="umap",
            n_neighbors=n_neighbors, min_dist=min_dist
        )

    @log_execution_time
    def visualize_concepts(self, df, show_evolution=False, title=None):
        """
        Unified visualization function for concepts
        
        Args:
            df: DataFrame with concepts and coordinates
            show_evolution: If True, shows concept evolution with transparency
                        based on message age
            title: Custom title (optional)
        
        Returns:
            Matplotlib figure
        """
        viz_logger.info(f"Creating concept visualization (evolution={show_evolution})")
        
        if df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No concepts to visualize",
                ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig
            
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Simplified color map
        colors = {
            "user": TEAL,
            "assistant": ORANGE,
        }
        
        # Determine if we're in chat mode
        is_chat_mode = MESSAGE_ID in df.columns or "message_id" in df.columns
        
        # Standardize column names for compatibility with existing code
        message_id_col = MESSAGE_ID if MESSAGE_ID in df.columns else "message_id"
        message_type_col = CONTENT_TYPE if CONTENT_TYPE in df.columns else "message_type"
        
        # Calculate alpha (transparency) based on message age if requested
        if show_evolution and message_id_col in df.columns:
            max_message_id = df[message_id_col].max()
            df[ALPHA] = df[message_id_col].apply(
                lambda mid: max(0.3, 1.0 - (max_message_id - mid) / max(1, max_message_id))
            )
        else:
            df[ALPHA] = 0.7  # Default alpha
        
        # Plot points
        # Simplify coloring logic
        if is_chat_mode and message_type_col in df.columns:
            # Chat mode - color by message type
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
            # Standard mode - color by source (user/assistant)
            source_col = "source"  # We don't change this for backward compatibility
            
            # Map values in "source" column to correct colors
            for source_value, group in df.groupby(source_col):
                # Simplify color decision
                color = colors.get(source_value, "gray")
                
                ax.scatter(
                    group[X_COORD] if X_COORD in df.columns else group["x"],
                    group[Y_COORD] if Y_COORD in df.columns else group["y"],
                    label=source_value,  # Use source_value directly as label
                    color=color,
                    alpha=0.7,
                    s=100
                )
            ax.legend()
        
        # Add labels to nodes
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
        
        # Add graph edges if available
        if self.concept_graph:
            # Create node position mapping
            node_positions = {}
            for _, row in df.iterrows():
                node_label = row[LABEL] if LABEL in df.columns else row["label"]
                x_coord = row[X_COORD] if X_COORD in df.columns else row["x"]
                y_coord = row[Y_COORD] if Y_COORD in df.columns else row["y"]
                node_positions[node_label] = (x_coord, y_coord)
            
            # Calculate alpha for nodes (for edges)
            node_alphas = {row[LABEL] if LABEL in df.columns else row["label"]: 
                        row.get(ALPHA, 0.7) for _, row in df.iterrows()}
            
            # Draw all edges
            for u, v, data in self.concept_graph.edges(data=True):
                if u in node_positions and v in node_positions:
                    # Use the smaller alpha of the two connected nodes
                    edge_alpha = min(node_alphas.get(u, 0.5), node_alphas.get(v, 0.5)) * 0.7
                    
                    # Different style for different edge types
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
        
        # Appropriate title for the mode
        if title:
            ax.set_title(title, fontsize=14)
        elif show_evolution or is_chat_mode:
            ax.set_title('Conceptual Evolution of Conversation', fontsize=14)
        else:
            ax.set_title('Conceptual Map of Transformation', fontsize=14)
        
        # Disable axes for a cleaner look
        ax.set_axis_off()
        plt.tight_layout()
        
        return fig

    @log_execution_time
    def visualize_concepts_interactive(self, df, show_evolution=False, title=None):
        """
        Creates an interactive visualization of concepts using Plotly
        
        Args:
            df: DataFrame with concepts and coordinates
            show_evolution: If True, shows concept evolution with transparency
                        based on message age
            title: Custom title (optional)
        
        Returns:
            Plotly figure object
        """

        def truncate_text(text, max_length=100):
            """Truncates long texts for tooltips"""
            if not text or len(text) <= max_length:
                return text
            return text[:max_length] + "..."
        
        viz_logger.info(f"Creating interactive concept visualization (evolution={show_evolution})")
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No concepts to visualize",
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(showlegend=False)
            return fig
        
        # Create figure
        fig = go.Figure()
        
        # Simplified color map
        colors = {
            "user": TEAL,
            "assistant": ORANGE,
        }
        
        # Determine if we're in chat mode
        is_chat_mode = MESSAGE_ID in df.columns or "message_id" in df.columns
        
        # Standardize column names for compatibility
        message_id_col = MESSAGE_ID if MESSAGE_ID in df.columns else "message_id"
        message_type_col = CONTENT_TYPE if CONTENT_TYPE in df.columns else "message_type"
        x_coord_col = X_COORD if X_COORD in df.columns else "x"
        y_coord_col = Y_COORD if Y_COORD in df.columns else "y"
        label_col = LABEL if LABEL in df.columns else "label"
        source_col = "source"  # For backward compatibility
        
        # Calculate alpha (transparency) based on message age if requested
        if show_evolution and message_id_col in df.columns:
            max_message_id = df[message_id_col].max()
            df[ALPHA] = df[message_id_col].apply(
                lambda mid: max(0.3, 1.0 - (max_message_id - mid) / max(1, max_message_id))
            )
        else:
            df[ALPHA] = 0.7  # Default alpha
        
        # Draw edges first if graph is available
        if self.concept_graph:
            # Create node position mapping
            node_positions = {}
            for _, row in df.iterrows():
                node_label = row[label_col]
                x_coord = row[x_coord_col]
                y_coord = row[y_coord_col]
                node_positions[node_label] = (x_coord, y_coord)
            
            # Calculate alpha for nodes (for edges)
            node_alphas = {row[label_col]: row.get(ALPHA, 0.7) for _, row in df.iterrows()}
            
            # Group edges by type for better styling
            edge_groups = {
                "standard": {"color": "#d9d9d9", "width": 0.8, "dash": "solid"},
                "same_message": {"color": "black", "width": 1.0, "dash": "solid"},
                "input-output": {"color": "green", "width": 1.2, "dash": "dash"}
            }
            
            for edge_type, style in edge_groups.items():
                x_edges, y_edges = [], []
                
                for u, v, data in self.concept_graph.edges(data=True):
                    if u not in node_positions or v not in node_positions:
                        continue
                    
                    # Check edge type
                    current_type = data.get("type", "standard")
                    if (edge_type == "standard" and current_type not in ["same_message", "input-output"]) or \
                    (edge_type != "standard" and current_type == edge_type):
                        
                        # Add coordinates for the line
                        x_edges.extend([node_positions[u][0], node_positions[v][0], None])
                        y_edges.extend([node_positions[u][1], node_positions[v][1], None])
                
                if x_edges:  # Only add trace if there are edges of this type
                    edge_trace = go.Scatter(
                        x=x_edges, 
                        y=y_edges,
                        mode='lines',
                        line=dict(
                            color=style["color"],
                            width=style["width"],
                            dash=style["dash"]
                        ),
                        opacity=0.6,
                        hoverinfo='none',
                        showlegend=False
                    )
                    fig.add_trace(edge_trace)
        
        # Plot nodes with hover information
        if is_chat_mode and message_type_col in df.columns:
            # Chat mode - color by message type
            for group_name, group_df in df.groupby(message_type_col):
                marker_color = colors.get(group_name, "gray")
                hover_text = [
                    f"<b>{row[label_col]}</b><br>" +
                    f"Category: {row.get('category', 'N/A')}<br>" +
                    f"Description: {truncate_text(row.get('description', 'N/A'))}<br>" +
                    f"Message ID: {row.get(message_id_col, 'N/A')}"
                    for _, row in group_df.iterrows()
                ]
                
                fig.add_trace(go.Scatter(
                    x=group_df[x_coord_col],
                    y=group_df[y_coord_col],
                    mode='markers+text',
                    marker=dict(
                        color=marker_color,
                        size=16,
                        opacity=group_df[ALPHA].tolist(),
                        line=dict(width=1, color='#888888')
                    ),
                    text=group_df[label_col],
                    textposition="top center",
                    name=group_name,
                    hovertext=hover_text,
                    hoverinfo='text',
                    hoverlabel=dict(
                        bgcolor='rgba(50, 50, 50, 0.9)',
                        bordercolor='#444444',
                        font=dict(size=12, color='white')
                    )
                ))
        else:
            # Standard mode - color by source (user/assistant)
            for group_name, group_df in df.groupby(source_col):
                marker_color = colors.get(group_name, "gray")
                
                hover_text = [
                    f"<b>{row[label_col]}</b><br>" +
                    f"Category: {row.get('category', 'N/A')}<br>" +
                    f"Description: {truncate_text(row.get('description', 'N/A'))}"
                    for _, row in group_df.iterrows()
                ]
                
                fig.add_trace(go.Scatter(
                    x=group_df[x_coord_col],
                    y=group_df[y_coord_col],
                    mode='markers+text',
                    marker=dict(
                        color=marker_color,
                        size=12,
                        opacity=group_df[ALPHA].tolist(),
                        line=dict(width=1, color='black')
                    ),
                    text=group_df[label_col],
                    textposition="top center",
                    name=group_name,
                    hovertext=hover_text,
                    hoverinfo='text',
                    hoverlabel=dict(
                        bgcolor='rgba(50, 50, 50, 0.9)',
                        bordercolor='#444444',
                        font=dict(size=12, color='white')
                    )
                ))
        
        # Set title
        title_text = title
        if title_text is None:
            if show_evolution or is_chat_mode:
                title_text = 'Conceptual Evolution of Conversation'
            else:
                title_text = 'Conceptual Map of Transformation'
        
        # Update layout
        fig.update_layout(
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#212021',
            paper_bgcolor='#212021',
            font=dict(color='white')
        )
        
        return fig

    @log_execution_time
    def visualize_static(self, df):
        """
        Wrapper for backward compatibility that calls the unified method
        """
        return self.visualize_concepts(df, show_evolution=False)

    @log_execution_time
    def visualize_static_chat(self, df):
        """
        Wrapper for backward compatibility that calls the unified method
        """
        return self.visualize_concepts(df, show_evolution=True)

    @log_execution_time
    def process_text_pair(self, input_text, output_text, use_llm=False,
                          name_weight=1.0, dim_reduction="umap",
                          use_mst=True, show_io_links=False):
        """
        Processes a pair of texts (input/output) and creates visualizations, with caching

        Args:
            input_text: Input text
            output_text: Output text
            use_llm: Whether to use LLM for concept extraction
            name_weight: Weight to give to concept name vs description (0.0-1.0)
            dim_reduction: Dimensional reduction method ("umap", "tsne", or "pca")
            use_mst: Whether to use Minimum Spanning Tree instead of k-NN
            show_io_links: Whether to show direct input-output connections

        Returns:
            DataFrame and figures
        """
        perf_logger.info(f"Processing text pair - Input length: {len(input_text)} characters, output: {len(output_text)} characters")
        
        try:
            # Generate cache keys for input and output
            params = {
                "use_llm": use_llm,
                "name_weight": name_weight
            }
            input_cache_key = self._get_cache_key(input_text, params)
            output_cache_key = self._get_cache_key(output_text, params)
            
            # Extract concepts (with cache)
            input_concepts = None
            if input_cache_key in self.concept_cache:
                perf_logger.info("Using input concepts from cache")
                input_concepts = self.concept_cache[input_cache_key]
            else:
                # Extract concepts normally
                if use_llm and self.llm_client:
                    input_concepts = self.extract_concepts_with_llm(input_text, True)
                else:
                    input_concepts = self.extract_concepts_alternative(input_text, True)
                
                # Calculate embeddings with name/description weight
                input_concepts = self.compute_embeddings(input_concepts, name_weight=name_weight)
                
                # Save in cache
                self.concept_cache[input_cache_key] = input_concepts
            
            # Repeat for output
            output_concepts = None
            if output_cache_key in self.concept_cache:
                perf_logger.info("Using output concepts from cache")
                output_concepts = self.concept_cache[output_cache_key]
            else:
                # Extract concepts normally
                if use_llm and self.llm_client:
                    output_concepts = self.extract_concepts_with_llm(output_text, False)
                else:
                    output_concepts = self.extract_concepts_alternative(output_text, False)
                
                # Calculate embeddings with name/description weight
                output_concepts = self.compute_embeddings(output_concepts, name_weight=name_weight)
                
                # Save in cache
                self.concept_cache[output_cache_key] = output_concepts
            
            # Combine concepts
            all_concepts = input_concepts + output_concepts
            
            # Generate key for the graph
            graph_params = {
                "use_mst": use_mst,
                "show_io_links": show_io_links,
                "input_key": input_cache_key,
                "output_key": output_cache_key
            }
            graph_cache_key = self._get_cache_key("graph", graph_params)
            
            # Build graph with caching
            if graph_cache_key in self.graph_cache:
                perf_logger.info("Using graph from cache")
                self.concept_graph = self.graph_cache[graph_cache_key]
            else:
                # Build graph normally
                self.concept_graph = self.build_concept_graph(all_concepts, use_mst=use_mst, show_io_links=show_io_links)
                # Save in cache
                self.graph_cache[graph_cache_key] = self.concept_graph
            
            # Generate key for dimensional reduction
            reduction_params = {
                "method": dim_reduction,
                "input_key": input_cache_key,
                "output_key": output_cache_key
            }
            reduction_cache_key = self._get_cache_key("reduction", reduction_params)
            
            # Reduce dimensionality with caching
            df = None
            if reduction_cache_key in self.reduction_cache:
                perf_logger.info(f"Using {dim_reduction} from cache")
                df = self.reduction_cache[reduction_cache_key]
            else:
                # Reduce dimensionality normally
                df = self.reduce_dimensions(all_concepts, method=dim_reduction)
                # Save in cache
                self.reduction_cache[reduction_cache_key] = df
            
            # Create static visualization
            fig_static = self.visualize_concepts_interactive(df)
            
            return df, fig_static, "Processing completed successfully (with optimized caching)"
            
        except Exception as e:
            print(f"Error in processing: {e}")
            traceback.print_exc()
            # Create empty dataframe and figures in case of error
            empty_df = pd.DataFrame(columns=["label", "category", "description", "source", "x", "y"])
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            
            return empty_df, fig, f"Error: {str(e)}"

    @log_execution_time
    def process_new_message(self, message, is_user=True, name_weight=1.0,
                            dim_reduction_method="umap", use_mst=True,
                            show_io_links=False, use_llm=False):
        """
        Processes a new message in the chat and updates the concept graph

        Args:
            message: Message text
            is_user: True if the message is from the user, False if from the system
            name_weight: Weight to give to concept name vs description (0.0-1.0)
            dim_reduction_method: Dimensional reduction method ("umap", "tsne", or "pca")
            use_mst: Whether to use Minimum Spanning Tree instead of k-NN
            show_io_links: Whether to show direct input-output connections
            use_llm: Whether to use LLM for concept extraction

        Returns:
            Updated DataFrame and figures
        """
        content_type = "user" if is_user else "assistant"
        chat_logger.info(f"Processing new message - {content_type} - Length: {len(message)} characters")

        try:
            # Increment counter and save message
            self.message_counter += 1
            message_id = self.message_counter

            self.chat_history.append({
                "id": message_id,
                "content": message,
                "is_user": is_user,
                "timestamp": time.time()
            })

            # Check if we've already calculated embeddings for this text
            if message in self.concept_cache:
                chat_logger.info("Using concepts from cache")
                new_concepts = self.concept_cache[message]
            else:
                # Extract concepts and calculate embeddings
                chat_logger.info("Extracting concepts from message")
                if use_llm and self.llm_client:
                    new_concepts = self.extract_concepts_with_llm(message, is_user=is_user)
                else:
                    new_concepts = self.extract_concepts_alternative(message, is_user=is_user)

                new_concepts = self.compute_embeddings(new_concepts, name_weight=name_weight)
                # Save in cache
                self.concept_cache[message] = new_concepts

            # Add message metadata
            for concept in new_concepts:
                concept[MESSAGE_ID] = message_id
                concept["message_type"] = content_type

            # Add new concepts to the overall list
            self.chat_concepts.extend(new_concepts)

            # Check safety limit on size
            if len(self.chat_concepts) > 500:  # Conservative limit
                return None, None, "Concept limit reached, reset the conversation"

            # Rebuild the graph with all concepts
            self.build_concept_graph(self.chat_concepts, use_mst=use_mst, show_io_links=show_io_links)

            # Reduce dimensionality with the chosen method
            df = self.reduce_dimensions(self.chat_concepts, method=dim_reduction_method)

            # Create visualization with the unified function
            fig_static = self.visualize_concepts_interactive(df, show_evolution=True)
            return df, fig_static, f"Message processed: found {len(new_concepts)} new concepts"

        except Exception as e:
            chat_logger.error(f"Error processing message: {e}", exc_info=True)
            # Create empty dataframe and figures in case of error
            empty_df = pd.DataFrame(columns=["label", "category", "description", "source", "x", "y"])
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', fontsize=12)
            ax.set_axis_off()

            return empty_df, fig, f"Error: {str(e)}"

    @log_execution_time
    def generate_chat_response(self, history):
        """
        Generates a response based on the conversation using OpenAI API
        without custom system message
        
        Args:
            history: Conversation history in format [{"role": "...", "content": "..."}]
            
        Returns:
            String with the response generated by the model
        """
        # If we don't have an OpenAI client or not enough messages
        if not self.llm_client or len(history) < 1:
            # Improved fallback response
            import random
            fallback_responses = [
                "I can't communicate with the model or you haven't provided enough messages.",
            ]
            return random.choice(fallback_responses)
    
        try:
            chat_logger.info("Generating response via OpenAI")
            
            # API call with only the existing conversation
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=history,
                temperature=0.7,
                max_tokens=150,
                timeout=10
            )
            
            # Extract the response
            return response.choices[0].message.content
            
        except Exception as e:
            chat_logger.error(f"Error generating response: {str(e)}")
            return "Error generating response"


# GRADIO INTERFACE
def create_gradio_interface():
    visualizer = ConceptTransformationVisualizer()
    visualizer.initialize_chat_mode()  # Initialize chat at startup

    # Load API key from .env if available
    default_api_key = os.environ.get("OPENAI_API_KEY", "")

    with gr.Blocks(title="Prompt Rover") as demo:
        gr.Markdown("# Prompt Rover")

        # Main horizontal layout
        with gr.Row():
            # Left column for input with tabs
            with gr.Column(scale=1):
                with gr.Tabs() as tabs:
                    # Tab for Input/Output mode
                    with gr.TabItem("Input/Output Mode"):
                        input_text = gr.Textbox(label="Input Text", placeholder="Enter input text", lines=5)
                        output_text = gr.Textbox(label="Output Text", placeholder="Enter output text", lines=10)
                        with gr.Accordion("Advanced options", open=False):
                            name_weight_io = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                                                   label="Weight of name vs. description")
                            dim_reduction_io = gr.Radio(["umap", "tsne", "pca"], label="Dimensionality reduction method", value="tsne")
                            use_mst_io = gr.Checkbox(label="Use Minimum Spanning Tree", value=True)
                            show_io_links_io = gr.Checkbox(label="Show input-output connections", value=True)
                            use_llm_io = gr.Checkbox(label="Use LLM for concept extraction", value=True)
                            openai_api_key_io = gr.Textbox(label="OpenAI API Key (optional)", value=default_api_key, placeholder="Enter if you want to use LLM")
                        submit_btn = gr.Button("Process and visualize")
                        example_btn = gr.Button("Load flower and wisdom example")

                    # Tab for Chat mode
                    with gr.TabItem("Chat Mode"):
                        chatbot = gr.Chatbot(height=400, label="Conversation", type="messages")  # Use type="messages" to avoid warning
                        chat_msg = gr.Textbox(placeholder="Write a message here...", label="Message")
                        with gr.Accordion("Advanced options", open=False):
                            name_weight_chat = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                                                   label="Weight of name vs. description")
                            dim_reduction_chat = gr.Radio(["umap", "tsne", "pca"], label="Dimensionality reduction method", value="tsne")
                            use_mst_chat = gr.Checkbox(label="Use Minimum Spanning Tree", value=True)
                            show_io_links_chat = gr.Checkbox(label="Show input-output connections", value=False)
                            use_llm_chat = gr.Checkbox(label="Use LLM for concept extraction", value=True)
                            openai_api_key_chat = gr.Textbox(label="OpenAI API Key (optional)", value=default_api_key, placeholder="Enter if you want to use LLM")
                        with gr.Row():
                            chat_submit = gr.Button("Send")
                            chat_clear = gr.Button("Clear conversation")
                            reinit_btn = gr.Button("Reinitialize system")

            # Right column for visualization and results
            with gr.Column(scale=2):
                static_output = gr.Plot(label="Concept visualization")
                df_output = gr.DataFrame(label="Extracted concepts")
                status = gr.Textbox(label="Status", interactive=False, value="System initialized and ready to use")

        # Function to reinitialize chat
        def reinit_chat():
            visualizer.initialize_chat_mode()
            return "System reinitialized"

        # Function to process chat messages
        def process_chat_message(message, history, name_weight, dim_reduction, use_mst, show_io_links, use_llm, openai_api_key):
            if not message.strip():
                return "", history, None, None, "Empty message"

            # Configure OpenAI API Key if provided and requested
            if use_llm and openai_api_key.strip():
                if not visualizer.initialize_openai_client(openai_api_key):
                    return "", history, None, None, "Error initializing OpenAI API"

            # Process user message
            try:
                if history is None:
                    history = []

                # For messages format
                history.append({"role": "user", "content": message})

                # Analyze user message
                df, fig_static, status_msg = visualizer.process_new_message(
                    message, is_user=True,
                    name_weight=name_weight,
                    dim_reduction_method=dim_reduction,
                    use_mst=use_mst,
                    show_io_links=show_io_links,
                    use_llm=use_llm
                )

                # Assistant response
                assistant_response = visualizer.generate_chat_response(history)
                history.append({"role": "assistant", "content": assistant_response})

                # Also analyze system response
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
                error_msg = f"Error during processing: {str(e)}"
                print(error_msg)
                traceback.print_exc()

                # Error handling
                if history is None:
                    history = []
                history.append({"role": "assistant", "content": "I'm sorry, an error occurred during processing"})
                return "", history, None, None, error_msg

        # Function to clear chat
        def clear_chat():
            visualizer.initialize_chat_mode()
            return "", [], None, None, "Conversation cleared"

        # Load example
        def load_example():
            input_text_example = "flower wisdom"
            output_text_example = """If we think about the symbolism of flowers in relation to wisdom, the lotus flower is one of the most representative. In the imagery of many cultures, especially Eastern ones, the lotus grows in mud but emerges pure and immaculate on the water's surface, symbolizing the capacity for spiritual and intellectual elevation, just like those who acquire wisdom through life's difficulties.\n
            Sage also has a name that recalls wisdom, deriving from the Latin salvus (safe, healthy). This plant, besides its healing properties, is often associated with prudence and experience.\n
            If we think instead of Western tradition, quercus robur, that is, the oak, although a tree and not a flower, is a symbol of wisdom for its longevity and its ability to withstand storms, just like a wise mind that knows how to face adversity without breaking.
            """
            return input_text_example, output_text_example

        def process_with_openai_key(input_text, output_text, use_llm, name_weight, dim_reduction, use_mst, show_io_links, openai_api_key):
            """Process text with API key initialization"""
            # If the user wants to use LLM, initialize the client
            if use_llm and openai_api_key.strip():
                visualizer.initialize_openai_client(openai_api_key)
            
            # Proceed with normal processing
            return visualizer.process_text_pair(
                input_text, output_text, use_llm, 
                name_weight, dim_reduction, 
                use_mst, show_io_links
            )


        # Links for Input/Output mode
        submit_btn.click(
            process_with_openai_key,
            inputs=[input_text, output_text, use_llm_io, name_weight_io, dim_reduction_io, use_mst_io, show_io_links_io, openai_api_key_io],
            outputs=[df_output, static_output, status]
        )
        example_btn.click(load_example, outputs=[input_text, output_text])

        # Links for Chat mode
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

        # Connect reinitialization button
        reinit_btn.click(
            reinit_chat,
            outputs=[status]
        )

    return demo

# Function to run in Colab
def run_in_colab():
    """
    Runs the visualizer in Google Colab environment
    """
    if IPYTHON_AVAILABLE:
        from IPython.display import display, HTML

        display(HTML(
            "<div style='background:#DDFFDD;border:1px solid #00CC00;padding:10px'>"
            "Interface launched! If it doesn't display automatically, check that it hasn't been blocked by your browser."
            "</div>"
        ))

        # Create and launch Gradio interface
        demo = create_gradio_interface()
        return demo.launch(inline=True, share=True, debug=True)
    else:
        print("IPython not available. Running in local environment")
        demo = create_gradio_interface()
        demo.launch(share=False, debug=True)

# Main script
if __name__ == "__main__":
    try:
        # Check if we're in Colab environment
        import google.colab
        # If no exception, we're in Colab
        print("Running in Google Colab environment")
        run_in_colab()
    except:
        # If exception, we're in local environment
        print("Running in local environment")
        demo = create_gradio_interface()
        demo.launch(share=False, debug=True)