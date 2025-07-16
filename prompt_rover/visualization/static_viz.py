"""
Modulo per visualizzazioni statiche con matplotlib
"""

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from typing import Optional, Dict, List, Tuple

from ..config import (
    TEAL, ORANGE, DEFAULT_FIGURE_SIZE, DEFAULT_SCATTER_SIZE, DEFAULT_ALPHA,
    X_COORD, Y_COORD, LABEL, ALPHA, MESSAGE_ID, CONTENT_TYPE
)
from ..utils import get_logger

logger = get_logger('visualization')

class StaticVisualizer:
    """
    Crea visualizzazioni statiche dei concetti usando matplotlib
    """
    
    def __init__(self):
        """Inizializza il visualizzatore statico"""
        self.colors = {
            "user": TEAL,
            "assistant": ORANGE,
            "input": TEAL,
            "output": ORANGE
        }
    
    def visualize_concepts(self, df: pd.DataFrame, 
                          graph: Optional[nx.Graph] = None,
                          show_evolution: bool = False,
                          title: Optional[str] = None,
                          figsize: Tuple[int, int] = DEFAULT_FIGURE_SIZE) -> plt.Figure:
        """
        Visualizza i concetti in modo statico
        
        Args:
            df: DataFrame con concetti e coordinate
            graph: Grafo NetworkX opzionale da visualizzare
            show_evolution: Se True, mostra evoluzione con trasparenza
            title: Titolo personalizzato
            figsize: Dimensione della figura
            
        Returns:
            Figura matplotlib
        """
        logger.info(f"Creazione visualizzazione statica (evolution={show_evolution})")
        
        if df.empty:
            return self._create_empty_figure(figsize)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determina se siamo in modalità chat
        is_chat_mode = self._is_chat_mode(df)
        
        # Calcola alpha se richiesto
        if show_evolution:
            self._calculate_alpha(df)
        else:
            df[ALPHA] = DEFAULT_ALPHA
        
        # Disegna punti
        self._draw_scatter_points(ax, df, is_chat_mode)
        
        # Aggiungi etichette
        self._add_labels(ax, df)
        
        # Disegna archi del grafo se disponibile
        if graph:
            self._draw_graph_edges(ax, df, graph)
        
        # Imposta titolo
        if title is None:
            title = self._get_default_title(show_evolution, is_chat_mode)
        ax.set_title(title, fontsize=14)
        
        # Rimuovi assi per un look più pulito
        ax.set_axis_off()
        plt.tight_layout()
        
        return fig
    
    def _create_empty_figure(self, figsize: Tuple[int, int]) -> plt.Figure:
        """Crea una figura vuota con messaggio"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Nessun concetto da visualizzare",
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    def _is_chat_mode(self, df: pd.DataFrame) -> bool:
        """Verifica se siamo in modalità chat"""
        return MESSAGE_ID in df.columns or "message_id" in df.columns
    
    def _calculate_alpha(self, df: pd.DataFrame):
        """Calcola valori alpha per evoluzione temporale"""
        message_id_col = MESSAGE_ID if MESSAGE_ID in df.columns else "message_id"
        
        if message_id_col in df.columns:
            max_message_id = df[message_id_col].max()
            df[ALPHA] = df[message_id_col].apply(
                lambda mid: max(0.3, 1.0 - (max_message_id - mid) / max(1, max_message_id))
            )
    
    def _draw_scatter_points(self, ax: plt.axes, df: pd.DataFrame, is_chat_mode: bool):
        """Disegna i punti scatter"""
        x_col = X_COORD if X_COORD in df.columns else "x"
        y_col = Y_COORD if Y_COORD in df.columns else "y"
        
        if is_chat_mode:
            # Modalità chat - colora per tipo messaggio
            message_type_col = CONTENT_TYPE if CONTENT_TYPE in df.columns else "message_type"
            
            for _, row in df.iterrows():
                alpha_value = row.get(ALPHA, DEFAULT_ALPHA)
                message_type = row.get(message_type_col, "")
                
                ax.scatter(
                    row[x_col], row[y_col],
                    color=self.colors.get(message_type, "gray"),
                    alpha=alpha_value,
                    s=DEFAULT_SCATTER_SIZE,
                    edgecolors='black',
                    linewidth=0.5
                )
        else:
            # Modalità standard - colora per source
            source_col = "source"
            
            for source_value, group in df.groupby(source_col):
                color = self.colors.get(source_value, "gray")
                
                ax.scatter(
                    group[x_col], group[y_col],
                    label=source_value,
                    color=color,
                    alpha=DEFAULT_ALPHA,
                    s=DEFAULT_SCATTER_SIZE
                )
            ax.legend()
    
    def _add_labels(self, ax: plt.axes, df: pd.DataFrame):
        """Aggiunge etichette ai punti"""
        x_col = X_COORD if X_COORD in df.columns else "x"
        y_col = Y_COORD if Y_COORD in df.columns else "y"
        label_col = LABEL if LABEL in df.columns else "label"
        
        for _, row in df.iterrows():
            alpha_value = row.get(ALPHA, DEFAULT_ALPHA)
            
            ax.annotate(
                row[label_col],
                (row[x_col], row[y_col]),
                fontsize=9,
                alpha=alpha_value,
                xytext=(5, 5),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=alpha_value)
            )
    
    def _draw_graph_edges(self, ax: plt.axes, df: pd.DataFrame, graph: nx.Graph):
        """Disegna gli archi del grafo"""
        # Crea mapping posizioni nodi
        x_col = X_COORD if X_COORD in df.columns else "x"
        y_col = Y_COORD if Y_COORD in df.columns else "y"
        label_col = LABEL if LABEL in df.columns else "label"
        
        node_positions = {
            row[label_col]: (row[x_col], row[y_col])
            for _, row in df.iterrows()
        }
        
        # Calcola alpha per nodi
        node_alphas = {
            row[label_col]: row.get(ALPHA, DEFAULT_ALPHA)
            for _, row in df.iterrows()
        }
        
        # Disegna archi
        for u, v, data in graph.edges(data=True):
            if u not in node_positions or v not in node_positions:
                continue
            
            # Usa l'alpha minore dei due nodi connessi
            edge_alpha = min(node_alphas.get(u, 0.5), node_alphas.get(v, 0.5)) * 0.7
            
            # Stile diverso per tipi di arco diversi
            edge_style = self._get_edge_style(data.get("type"))
            
            x = [node_positions[u][0], node_positions[v][0]]
            y = [node_positions[u][1], node_positions[v][1]]
            
            ax.plot(
                x, y,
                linestyle=edge_style["linestyle"],
                color=edge_style["color"],
                alpha=edge_alpha,
                linewidth=edge_style["linewidth"]
            )
    
    def _get_edge_style(self, edge_type: Optional[str]) -> Dict[str, any]:
        """Ottiene lo stile per un tipo di arco"""
        styles = {
            "same_message": {
                "linestyle": "-",
                "color": "purple", 
                "linewidth": 1.0
            },
            "input-output": {
                "linestyle": "--",
                "color": "green",
                "linewidth": 1.2
            },
            "mst": {
                "linestyle": "-",
                "color": "black",
                "linewidth": 0.8
            },
            "knn": {
                "linestyle": "-",
                "color": "black",
                "linewidth": 0.8
            }
        }
        
        return styles.get(edge_type, {
            "linestyle": "-",
            "color": "black",
            "linewidth": 0.8
        })
    
    def _get_default_title(self, show_evolution: bool, is_chat_mode: bool) -> str:
        """Ottiene il titolo di default"""
        if show_evolution or is_chat_mode:
            return "Evoluzione Concettuale della Conversazione"
        else:
            return "Mappa Concettuale della Trasformazione" 