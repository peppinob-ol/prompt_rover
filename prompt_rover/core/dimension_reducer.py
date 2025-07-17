"""
Modulo per la riduzione dimensionale degli embeddings
"""

import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List, Dict, Optional, Tuple

from ..config import (
    LABEL, CATEGORY, DESCRIPTION, CONTENT_TYPE, EMBEDDING, MESSAGE_ID,
    X_COORD, Y_COORD, UMAP_DEFAULT_NEIGHBORS, UMAP_DEFAULT_MIN_DIST,
    TSNE_DEFAULT_PERPLEXITY
)
from ..utils import get_logger, log_execution_time

logger = get_logger('dimension_reduction')

class DimensionReducer:
    """
    Gestisce la riduzione dimensionale degli embeddings per la visualizzazione
    """
    
    def __init__(self):
        """Inizializza il riduttore dimensionale"""
        self.umap_reducer = None
        self.tsne_reducer = None
        self.pca_reducer = None
        self.last_method = None
        
    @log_execution_time
    def reduce_dimensions(self, concepts: List[Dict], 
                         method: str = "umap",
                         n_neighbors: int = UMAP_DEFAULT_NEIGHBORS,
                         min_dist: float = UMAP_DEFAULT_MIN_DIST,
                         perplexity: int = TSNE_DEFAULT_PERPLEXITY) -> pd.DataFrame:
        """
        Riduce la dimensionalità degli embeddings dei concetti
        
        Args:
            concepts: Lista di concetti con embeddings
            method: Metodo di riduzione ('umap', 'tsne', o 'pca')
            n_neighbors: Parametro n_neighbors per UMAP
            min_dist: Parametro min_dist per UMAP
            perplexity: Parametro perplexity per t-SNE
            
        Returns:
            DataFrame con concetti e coordinate 2D
        """
        logger.info(f"Riduzione dimensionale con {method} per {len(concepts)} concetti")
        
        if len(concepts) == 0:
            return self._create_empty_dataframe()
        
        # Estrai embeddings
        embeddings = np.array([concept[EMBEDDING] for concept in concepts])
        
        # Gestisci caso di un singolo concetto
        if len(concepts) == 1:
            reduced_coords = np.array([[0.0, 0.0]])
        else:
            # Applica il metodo di riduzione scelto
            reduced_coords = self._apply_reduction_method(
                embeddings, method, n_neighbors, min_dist, perplexity
            )
        
        # Crea DataFrame con risultati
        return self._create_results_dataframe(concepts, reduced_coords)
    
    def _apply_reduction_method(self, embeddings: np.ndarray, 
                               method: str,
                               n_neighbors: int,
                               min_dist: float,
                               perplexity: int) -> np.ndarray:
        """
        Applica il metodo di riduzione dimensionale specificato
        
        Args:
            embeddings: Array di embeddings
            method: Metodo da utilizzare
            n_neighbors: Parametro per UMAP
            min_dist: Parametro per UMAP
            perplexity: Parametro per t-SNE
            
        Returns:
            Coordinate ridotte 2D
        """
        method_lower = method.lower()
        
        if method_lower == "umap" and len(embeddings) >= 4:
            return self._apply_umap(embeddings, n_neighbors, min_dist)
        elif method_lower == "tsne" and len(embeddings) >= 3:
            return self._apply_tsne(embeddings, perplexity)
        else:
            # Default a PCA o per dataset piccoli
            return self._apply_pca(embeddings)
    
    def _apply_umap(self, embeddings: np.ndarray, 
                   n_neighbors: int, 
                   min_dist: float) -> np.ndarray:
        """
        Applica riduzione UMAP
        
        Args:
            embeddings: Array di embeddings
            n_neighbors: Numero di vicini
            min_dist: Distanza minima
            
        Returns:
            Coordinate 2D
        """
        # Aggiusta n_neighbors se necessario
        adjusted_neighbors = min(n_neighbors, len(embeddings) - 1)
        
        # Riusa reducer esistente se compatibile
        if (self.umap_reducer is not None and 
            self.last_method == "umap" and
            hasattr(self.umap_reducer, 'n_neighbors') and 
            self.umap_reducer.n_neighbors == adjusted_neighbors):
            
            logger.info("Riutilizzo del reducer UMAP esistente")
            try:
                # Prova a usare transform
                reduced_coords = self.umap_reducer.transform(embeddings)
            except Exception:
                # Altrimenti, refit
                reduced_coords = self.umap_reducer.fit_transform(embeddings)
        else:
            logger.info("Creazione nuovo reducer UMAP")
            self.umap_reducer = UMAP(
                n_neighbors=adjusted_neighbors,
                min_dist=min_dist,
                n_components=2,
                metric='cosine',
                random_state=42
            )
            reduced_coords = self.umap_reducer.fit_transform(embeddings)
            self.last_method = "umap"
        
        return reduced_coords
    
    def _apply_tsne(self, embeddings: np.ndarray, perplexity: int) -> np.ndarray:
        """
        Applica riduzione t-SNE
        
        Args:
            embeddings: Array di embeddings
            perplexity: Parametro perplexity
            
        Returns:
            Coordinate 2D
        """
        # Aggiusta perplexity se necessario
        adjusted_perplexity = min(perplexity, len(embeddings) // 3)
        adjusted_perplexity = max(5, adjusted_perplexity)  # Minimo 5
        
        logger.info(f"Applicazione t-SNE con perplexity={adjusted_perplexity}")
        
        self.tsne_reducer = TSNE(
            n_components=2,
            perplexity=adjusted_perplexity,
            random_state=42,
            init='pca',
            learning_rate='auto'
        )
        reduced_coords = self.tsne_reducer.fit_transform(embeddings)
        self.last_method = "tsne"
        
        return reduced_coords
    
    def _apply_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Applica riduzione PCA
        
        Args:
            embeddings: Array di embeddings
            
        Returns:
            Coordinate 2D
        """
        n_components = min(2, len(embeddings))
        
        # Riusa reducer PCA se esiste
        if self.pca_reducer is not None and self.last_method == "pca":
            try:
                reduced_coords = self.pca_reducer.transform(embeddings)
            except:
                self.pca_reducer = PCA(n_components=n_components)
                self.pca_reducer.fit(embeddings)
                reduced_coords = self.pca_reducer.transform(embeddings)
        else:
            logger.info("Creazione nuovo reducer PCA")
            self.pca_reducer = PCA(n_components=n_components)
            self.pca_reducer.fit(embeddings)
            reduced_coords = self.pca_reducer.transform(embeddings)
            self.last_method = "pca"
        
        # Se abbiamo solo 2 concetti, allarghiamo artificialmente per visibilità
        if len(embeddings) == 2:
            reduced_coords = reduced_coords * 2
        
        return reduced_coords
    
    def _create_empty_dataframe(self) -> pd.DataFrame:
        """
        Crea un DataFrame vuoto con le colonne corrette
        
        Returns:
            DataFrame vuoto
        """
        return pd.DataFrame(columns=[
            LABEL, CATEGORY, DESCRIPTION, "source", X_COORD, Y_COORD
        ])
    
    def _create_results_dataframe(self, concepts: List[Dict], 
                                 reduced_coords: np.ndarray) -> pd.DataFrame:
        """
        Crea il DataFrame dei risultati con concetti e coordinate
        
        Args:
            concepts: Lista di concetti
            reduced_coords: Coordinate 2D ridotte
            
        Returns:
            DataFrame con risultati
        """
        # Crea DataFrame base
        df = pd.DataFrame({
            LABEL: [concept[LABEL] for concept in concepts],
            CATEGORY: [concept.get(CATEGORY, "") for concept in concepts],
            DESCRIPTION: [concept.get(DESCRIPTION, "") for concept in concepts],
            "source": [concept.get(CONTENT_TYPE, "") for concept in concepts],
            X_COORD: reduced_coords[:, 0],
            Y_COORD: reduced_coords[:, 1]
        })
        
        # Aggiungi colonne specifiche per modalità chat se presenti
        if any(MESSAGE_ID in concept for concept in concepts):
            df[MESSAGE_ID] = [concept.get(MESSAGE_ID, 0) for concept in concepts]
            df["message_type"] = [concept.get("message_type", "") for concept in concepts]
        
        return df
    
    def get_explained_variance(self) -> Optional[float]:
        """
        Ottiene la varianza spiegata (solo per PCA)
        
        Returns:
            Varianza spiegata se disponibile, None altrimenti
        """
        if self.last_method == "pca" and self.pca_reducer is not None:
            return float(sum(self.pca_reducer.explained_variance_ratio_))
        return None
    
    def reset(self):
        """Resetta tutti i reducer salvati"""
        self.umap_reducer = None
        self.tsne_reducer = None
        self.pca_reducer = None
        self.last_method = None
        logger.info("Reducer dimensionali resettati") 