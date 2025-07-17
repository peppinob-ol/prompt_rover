"""
Modulo per la costruzione di grafi concettuali
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Optional

from ..config import (
    LABEL, CATEGORY, DESCRIPTION, CONTENT_TYPE, EMBEDDING, MESSAGE_ID,
    DEFAULT_K_NEIGHBORS, DEFAULT_USE_MST, DEFAULT_SHOW_IO_LINKS
)
from ..utils import get_logger, log_execution_time

logger = get_logger('graph')

class GraphBuilder:
    """
    Costruisce grafi basati sulla similarità tra concetti
    """
    
    def __init__(self):
        """Inizializza il costruttore di grafi"""
        self.current_graph = None
        
    @log_execution_time
    def build_graph(self, concepts: List[Dict], 
                   k: int = DEFAULT_K_NEIGHBORS,
                   use_mst: bool = DEFAULT_USE_MST,
                   show_io_links: bool = DEFAULT_SHOW_IO_LINKS) -> nx.Graph:
        """
        Costruisce un grafo concettuale (k-NN o MST)
        
        Args:
            concepts: Lista di concetti con embeddings
            k: Numero di vicini più prossimi da connettere (per k-NN)
            use_mst: Se True, crea un Minimum Spanning Tree invece di k-NN
            show_io_links: Se True, mostra connessioni dirette tra concetti input e output
            
        Returns:
            Grafo NetworkX
        """
        logger.info(f"Costruzione grafo con {'MST' if use_mst else 'k-NN'} per {len(concepts)} concetti")
        
        if len(concepts) <= 1:
            return self._create_trivial_graph(concepts)
        
        # Crea grafo vuoto
        G = nx.Graph()
        
        # Controlla se siamo in modalità chat
        is_chat_mode = self._is_chat_mode(concepts)
        if is_chat_mode:
            logger.info("Modalità chat rilevata nella costruzione del grafo")
        
        # Aggiungi nodi con attributi
        self._add_nodes(G, concepts, is_chat_mode)
        
        # Costruisci connessioni
        if use_mst:
            self._build_mst(G, concepts)
        else:
            self._build_knn(G, concepts, k)
        
        # Aggiungi connessioni esplicite input-output se richiesto
        if show_io_links:
            self._add_io_links(G, concepts)
        
        # Se siamo in modalità chat, aggiungi connessioni intra-messaggio
        if is_chat_mode:
            self._add_intramessage_links(G, concepts)
        
        self.current_graph = G
        return G
    
    def _create_trivial_graph(self, concepts: List[Dict]) -> nx.Graph:
        """
        Crea un grafo vuoto o con un solo nodo
        
        Args:
            concepts: Lista di concetti (0 o 1 elemento)
            
        Returns:
            Grafo NetworkX
        """
        G = nx.Graph()
        if len(concepts) == 1:
            G.add_node(
                concepts[0][LABEL],
                category=concepts[0].get(CATEGORY, ""),
                description=concepts[0].get(DESCRIPTION, ""),
                source=concepts[0].get(CONTENT_TYPE, "")
            )
        self.current_graph = G
        return G
    
    def _is_chat_mode(self, concepts: List[Dict]) -> bool:
        """
        Verifica se siamo in modalità chat
        
        Args:
            concepts: Lista di concetti
            
        Returns:
            True se in modalità chat
        """
        return any(MESSAGE_ID in concept for concept in concepts)
    
    def _add_nodes(self, G: nx.Graph, concepts: List[Dict], is_chat_mode: bool):
        """
        Aggiunge nodi al grafo con attributi
        
        Args:
            G: Grafo NetworkX
            concepts: Lista di concetti
            is_chat_mode: Se siamo in modalità chat
        """
        for concept in concepts:
            node_attrs = {
                CATEGORY: concept.get(CATEGORY, ""),
                DESCRIPTION: concept.get(DESCRIPTION, ""),
                CONTENT_TYPE: concept.get(CONTENT_TYPE, ""),
                EMBEDDING: concept[EMBEDDING]
            }
            
            # Aggiungi attributi specifici per modalità chat
            if is_chat_mode and MESSAGE_ID in concept:
                node_attrs[MESSAGE_ID] = concept[MESSAGE_ID]
                node_attrs["message_type"] = concept.get("message_type", "")
            
            G.add_node(concept[LABEL], **node_attrs)
    
    def _build_mst(self, G: nx.Graph, concepts: List[Dict]):
        """
        Costruisce un Minimum Spanning Tree
        
        Args:
            G: Grafo NetworkX
            concepts: Lista di concetti
        """
        # Crea grafo completo con distanze
        G_complete = nx.Graph()
        
        # Aggiungi nodi
        labels = [concept[LABEL] for concept in concepts]
        for label in labels:
            G_complete.add_node(label)
        
        # Aggiungi archi con pesi basati sulla distanza
        embeddings = np.array([concept[EMBEDDING] for concept in concepts])
        
        for i, label_i in enumerate(labels):
            for j, label_j in enumerate(labels):
                if i < j:  # Evita duplicati e self-connections
                    # Calcola distanza euclidea come peso
                    dist = np.linalg.norm(embeddings[i] - embeddings[j])
                    G_complete.add_edge(label_i, label_j, weight=dist)
        
        # Crea MST
        mst = nx.minimum_spanning_tree(G_complete)
        
        # Copia archi MST nel grafo principale
        for u, v, data in mst.edges(data=True):
            # Inverti peso per avere similarità invece di distanza
            similarity = 1.0 / (data["weight"] + 0.1)  # Evita divisione per zero
            G.add_edge(u, v, weight=similarity, type="mst")
    
    def _build_knn(self, G: nx.Graph, concepts: List[Dict], k: int):
        """
        Costruisce un grafo k-NN
        
        Args:
            G: Grafo NetworkX
            concepts: Lista di concetti
            k: Numero di vicini più prossimi
        """
        labels = [concept[LABEL] for concept in concepts]
        embeddings = np.array([concept[EMBEDDING] for concept in concepts])
        
        # Normalizza embeddings per calcolo similarità coseno
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        
        # Calcola matrice di similarità
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Per ogni nodo, aggiungi archi ai k nodi più simili
        for i, label in enumerate(labels):
            # Ottieni indici dei k concetti più simili (escludendo self)
            similarities = similarity_matrix[i]
            similarities[i] = -1  # Escludi self
            top_k_indices = np.argsort(similarities)[-k:]
            
            # Aggiungi archi
            for idx in top_k_indices:
                if similarities[idx] > 0:  # Verifica che ci sia similarità positiva
                    G.add_edge(
                        label,
                        labels[idx],
                        weight=float(similarities[idx]),
                        type="knn"
                    )
    
    def _add_io_links(self, G: nx.Graph, concepts: List[Dict]):
        """
        Aggiunge connessioni esplicite tra concetti input e output
        
        Args:
            G: Grafo NetworkX
            concepts: Lista di concetti
        """
        input_concepts = [c for c in concepts if c.get(CONTENT_TYPE, "") == "input"]
        output_concepts = [c for c in concepts if c.get(CONTENT_TYPE, "") == "output"]
        
        for input_concept in input_concepts:
            input_emb = input_concept[EMBEDDING]
            input_label = input_concept[LABEL]
            
            for output_concept in output_concepts:
                output_emb = output_concept[EMBEDDING]
                output_label = output_concept[LABEL]
                
                # Calcola similarità coseno
                sim = np.dot(input_emb, output_emb) / (
                    np.linalg.norm(input_emb) * np.linalg.norm(output_emb)
                )
                
                # Aggiungi arco se similarità è alta
                if sim > 0.7:
                    G.add_edge(
                        input_label,
                        output_label,
                        weight=float(sim),
                        type="input-output"
                    )
    
    def _add_intramessage_links(self, G: nx.Graph, concepts: List[Dict]):
        """
        Aggiunge connessioni tra concetti dello stesso messaggio
        
        Args:
            G: Grafo NetworkX
            concepts: Lista di concetti
        """
        logger.info("Aggiunta connessioni intra-messaggio")
        
        # Raggruppa concetti per messaggio
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
                        G.add_edge(
                            label1, 
                            label2, 
                            weight=0.9, 
                            type="same_message"
                        )
        
        logger.info(f"Aggiunte connessioni per {len(message_groups)} messaggi")
    
    def get_node_importance(self, node: str) -> float:
        """
        Calcola l'importanza di un nodo nel grafo
        
        Args:
            node: Nome del nodo
            
        Returns:
            Score di importanza (0-1)
        """
        if not self.current_graph or node not in self.current_graph:
            return 0.0
        
        # Usa centralità di grado normalizzata
        degree_centrality = nx.degree_centrality(self.current_graph)
        return degree_centrality.get(node, 0.0)
    
    def get_communities(self) -> List[List[str]]:
        """
        Identifica comunità di concetti nel grafo
        
        Returns:
            Lista di comunità (liste di nodi)
        """
        if not self.current_graph or len(self.current_graph) == 0:
            return []
        
        # Usa l'algoritmo di Louvain per rilevare comunità
        communities = nx.community.louvain_communities(
            self.current_graph, 
            seed=42
        )
        
        return [list(community) for community in communities] 