"""
Modulo per visualizzazioni interattive con Plotly
"""

import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from typing import Optional, Dict, List

from ..config import (
    TEAL, ORANGE, DEFAULT_ALPHA,
    X_COORD, Y_COORD, LABEL, ALPHA, MESSAGE_ID, CONTENT_TYPE
)
from ..utils import get_logger

logger = get_logger('visualization')

class InteractiveVisualizer:
    """
    Crea visualizzazioni interattive dei concetti usando Plotly
    """
    
    def __init__(self):
        """Inizializza il visualizzatore interattivo"""
        self.colors = {
            "user": TEAL,
            "assistant": ORANGE,
            "input": TEAL,
            "output": ORANGE
        }
    
    def visualize_concepts(self, df: pd.DataFrame,
                          graph: Optional[nx.Graph] = None,
                          show_evolution: bool = False,
                          title: Optional[str] = None) -> go.Figure:
        """
        Crea una visualizzazione interattiva dei concetti
        
        Args:
            df: DataFrame con concetti e coordinate
            graph: Grafo NetworkX opzionale
            show_evolution: Se True, mostra evoluzione con trasparenza
            title: Titolo personalizzato
            
        Returns:
            Figura Plotly
        """
        logger.info(f"Creazione visualizzazione interattiva (evolution={show_evolution})")
        
        if df.empty:
            return self._create_empty_figure()
        
        # Crea figura
        fig = go.Figure()
        
        # Prepara i dati
        df = self._prepare_dataframe(df, show_evolution)
        
        # Determina modalità
        is_chat_mode = self._is_chat_mode(df)
        
        # Disegna archi del grafo se disponibile
        if graph:
            self._add_graph_edges(fig, df, graph)
        
        # Aggiungi nodi
        self._add_nodes(fig, df, is_chat_mode)
        
        # Configura layout
        self._configure_layout(fig, title, show_evolution, is_chat_mode)
        
        return fig
    
    def _create_empty_figure(self) -> go.Figure:
        """Crea una figura vuota"""
        fig = go.Figure()
        fig.add_annotation(
            text="No concepts to visualize",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(showlegend=False)
        return fig
    
    def _prepare_dataframe(self, df: pd.DataFrame, show_evolution: bool) -> pd.DataFrame:
        """Prepara il DataFrame per la visualizzazione"""
        df = df.copy()
        
        # Calcola alpha se necessario
        if show_evolution and self._is_chat_mode(df):
            message_id_col = MESSAGE_ID if MESSAGE_ID in df.columns else "message_id"
            max_message_id = df[message_id_col].max()
            df[ALPHA] = df[message_id_col].apply(
                lambda mid: max(0.3, 1.0 - (max_message_id - mid) / max(1, max_message_id))
            )
        else:
            df[ALPHA] = DEFAULT_ALPHA
        
        return df
    
    def _is_chat_mode(self, df: pd.DataFrame) -> bool:
        """Verifica se siamo in modalità chat"""
        return MESSAGE_ID in df.columns or "message_id" in df.columns
    
    def _add_graph_edges(self, fig: go.Figure, df: pd.DataFrame, graph: nx.Graph):
        """Aggiunge gli archi del grafo alla figura"""
        # Crea mapping posizioni
        x_col = X_COORD if X_COORD in df.columns else "x"
        y_col = Y_COORD if Y_COORD in df.columns else "y"
        label_col = LABEL if LABEL in df.columns else "label"
        
        node_positions = {
            row[label_col]: (row[x_col], row[y_col])
            for _, row in df.iterrows()
        }
        
        # Raggruppa archi per tipo
        edge_groups = {
            "standard": {"color": "#d9d9d9", "width": 0.8, "dash": "solid"},
            "same_message": {"color": "black", "width": 1.0, "dash": "solid"},
            "input-output": {"color": "green", "width": 1.2, "dash": "dash"}
        }
        
        for edge_type, style in edge_groups.items():
            x_edges, y_edges = [], []
            
            for u, v, data in graph.edges(data=True):
                if u not in node_positions or v not in node_positions:
                    continue
                
                # Verifica tipo arco
                current_type = data.get("type", "standard")
                if (edge_type == "standard" and current_type not in ["same_message", "input-output"]) or \
                   (edge_type != "standard" and current_type == edge_type):
                    
                    # Aggiungi coordinate per la linea
                    x_edges.extend([node_positions[u][0], node_positions[v][0], None])
                    y_edges.extend([node_positions[u][1], node_positions[v][1], None])
            
            if x_edges:  # Aggiungi trace solo se ci sono archi di questo tipo
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
    
    def _add_nodes(self, fig: go.Figure, df: pd.DataFrame, is_chat_mode: bool):
        """Aggiunge i nodi alla figura"""
        # Identifica colonne
        x_col = X_COORD if X_COORD in df.columns else "x"
        y_col = Y_COORD if Y_COORD in df.columns else "y"
        label_col = LABEL if LABEL in df.columns else "label"
        
        if is_chat_mode:
            # Modalità chat
            message_type_col = CONTENT_TYPE if CONTENT_TYPE in df.columns else "message_type"
            
            for group_name, group_df in df.groupby(message_type_col):
                self._add_node_group(
                    fig, group_df, group_name,
                    x_col, y_col, label_col,
                    is_chat_mode=True
                )
        else:
            # Modalità standard
            source_col = "source"
            
            for group_name, group_df in df.groupby(source_col):
                self._add_node_group(
                    fig, group_df, group_name,
                    x_col, y_col, label_col,
                    is_chat_mode=False
                )
    
    def _add_node_group(self, fig: go.Figure, group_df: pd.DataFrame,
                       group_name: str, x_col: str, y_col: str, label_col: str,
                       is_chat_mode: bool = False):
        """Aggiunge un gruppo di nodi"""
        marker_color = self.colors.get(group_name, "gray")
        
        # Prepara hover text
        hover_text = []
        for _, row in group_df.iterrows():
            text = f"<b>{row[label_col]}</b><br>"
            text += f"Category: {row.get('category', 'N/A')}<br>"
            text += f"Description: {self._truncate_text(row.get('description', 'N/A'))}"
            
            if is_chat_mode and MESSAGE_ID in row:
                text += f"<br>ID Messaggio: {row.get(MESSAGE_ID, 'N/A')}"
            
            hover_text.append(text)
        
        # Crea trace separati per i nodi (marker) e per le etichette (testo).
        # Questo evita problemi di rendering su alcune versioni di Plotly/Kaleido
        # che possono ignorare i marker quando si usa "markers+text" in combinazione
        # con opacità vettoriale.

        # Trace per i marker
        marker_trace = go.Scatter(
            x=group_df[x_col],
            y=group_df[y_col],
            mode='markers',
            marker=dict(
                # Applichiamo l'alpha per-punto convertendo il colore HEX in RGBA
                color=[self._hex_to_rgba(marker_color, a) for a in group_df[ALPHA].tolist()],
                size=16 if is_chat_mode else 12,
                line=dict(width=1, color='#888888' if is_chat_mode else 'black')
            ),
            name=group_name,
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=True
        )

        # Trace per le etichette testo (senza marker)
        text_trace = go.Scatter(
            x=group_df[x_col],
            y=group_df[y_col],
            mode='text',
            text=group_df[label_col],
            textposition="top center",
            textfont=dict(color='white'),
            hoverinfo='none',
            showlegend=False
        )

        # Aggiunge i trace alla figura
        fig.add_trace(marker_trace)
        fig.add_trace(text_trace)
    
    def _truncate_text(self, text: str, max_length: int = 100) -> str:
        """Tronca testi lunghi per i tooltip"""
        if not text or len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    @staticmethod
    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        """Converte colore HEX in stringa RGBA con alpha specifico"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return hex_color  # fallback: restituisce valore originale
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r}, {g}, {b}, {alpha})'
    
    def _configure_layout(self, fig: go.Figure, title: Optional[str],
                         show_evolution: bool, is_chat_mode: bool):
        """Configura il layout della figura"""
        # Determina titolo
        if title is None:
            if show_evolution or is_chat_mode:
                title = 'Drift of Conversation Concepts'
            else:
                title = 'Map of concept drift'
        
        # Aggiorna layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#212021',
            paper_bgcolor='#212021',
            font=dict(color='white')
        ) 