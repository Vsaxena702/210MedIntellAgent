"""
Medical Knowledge Graph Visualizer
Creates interactive Plotly visualizations from NetworkX graphs
"""

import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MedicalGraphVisualizer:
    """Create interactive Plotly visualizations of medical knowledge graphs"""
    
    # Medical entity colors (consistent with graph builder)
    MEDICAL_COLORS = {
        'patient': '#9b59b6',      # Purple - central
        'condition': '#e74c3c',    # Red - serious
        'medication': '#3498db',   # Blue - treatment
        'procedure': '#2ecc71',    # Green - action
        'observation': '#f39c12',  # Orange - data
        'immunization': '#1abc9c', # Cyan - prevention
        'encounter': '#f1c40f',    # Yellow - event
        'provider': '#95a5a6',     # Gray - person
        'unknown': '#34495e'       # Dark gray - unclassified
    }
    
    # Emoji icons for entity types
    ENTITY_ICONS = {
        'patient': '👤',
        'condition': '🔴',
        'medication': '💊',
        'procedure': '⚕️',
        'observation': '📊',
        'immunization': '💉',
        'encounter': '🏥',
        'provider': '👨‍⚕️',
        'unknown': '❓'
    }
    
    def __init__(self, width: int = 1200, height: int = 800):
        """
        Initialize the visualizer
        
        Args:
            width: Plot width in pixels
            height: Plot height in pixels
        """
        self.width = width
        self.height = height
    
    def visualize(self, graph: nx.DiGraph, title: str = "Medical Knowledge Graph") -> str:
        """
        Create interactive Plotly visualization
        
        Args:
            graph: NetworkX directed graph with medical metadata
            title: Title for the visualization
            
        Returns:
            HTML string containing the interactive plot
        """
        if graph.number_of_nodes() == 0:
            return self._create_empty_plot_html(title)
        
        try:
            # Calculate layout
            pos = self._calculate_layout(graph)
            
            # Create edge traces
            edge_traces = self._create_edge_traces(graph, pos)
            
            # Create node trace
            node_trace = self._create_node_trace(graph, pos)
            
            # Create figure
            fig = self._create_figure(edge_traces, node_trace, title)
            
            # Convert to HTML
            html = fig.to_html(
                include_plotlyjs='cdn',
                full_html=False,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'medical_knowledge_graph',
                        'height': self.height,
                        'width': self.width,
                        'scale': 2
                    }
                }
            )
            
            logger.info(f"Visualization created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            return html
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return self._create_error_plot_html(str(e))
    
    def _calculate_layout(self, graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """
        Calculate node positions using spring layout
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary mapping node -> (x, y) position
        """
        # Use spring layout for organic appearance
        # k parameter controls spacing (higher = more space)
        pos = nx.spring_layout(
            graph,
            k=2.5,  # Spacing between nodes
            iterations=50,  # Layout iterations
            seed=42  # For reproducibility
        )
        
        # Scale positions to fit nicely in plot
        scale_factor = 10
        pos = {node: (x * scale_factor, y * scale_factor) for node, (x, y) in pos.items()}
        
        return pos
    
    def _create_edge_traces(self, graph: nx.DiGraph, pos: Dict) -> List[go.Scatter]:
        """
        Create Plotly traces for edges with arrows
        
        Args:
            graph: NetworkX graph
            pos: Node positions
            
        Returns:
            List of Plotly Scatter traces for edges
        """
        edge_traces = []
        
        for source, target, data in graph.edges(data=True):
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            # Get edge label
            label = data.get('label', data.get('relationship', ''))
            
            # Create edge line
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=2,
                    color='#95a5a6',  # Gray
                    dash='solid'
                ),
                hoverinfo='text',
                hovertext=f"{label}: {source} → {target}",
                showlegend=False
            )
            edge_traces.append(edge_trace)
            
            # Add arrow annotation
            # Calculate arrow position (90% along the edge)
            arrow_x = x0 + 0.9 * (x1 - x0)
            arrow_y = y0 + 0.9 * (y1 - y0)
            
            # Add edge label at midpoint
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            
            edge_label_trace = go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode='text',
                text=[label],
                textposition='middle center',
                textfont=dict(size=9, color='#7f8c8d'),
                hoverinfo='skip',
                showlegend=False
            )
            edge_traces.append(edge_label_trace)
        
        return edge_traces
    
    def _create_node_trace(self, graph: nx.DiGraph, pos: Dict) -> go.Scatter:
        """
        Create Plotly trace for nodes
        
        Args:
            graph: NetworkX graph
            pos: Node positions
            
        Returns:
            Plotly Scatter trace for nodes
        """
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        node_hover_text = []
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node attributes
            node_data = graph.nodes[node]
            node_type = node_data.get('type', 'unknown')
            node_label = node_data.get('label', str(node))
            node_size = node_data.get('size', 20)
            centrality = node_data.get('centrality', 0)
            
            # Color by type
            color = node_data.get('color', self.MEDICAL_COLORS.get(node_type, '#34495e'))
            node_colors.append(color)
            
            # Size by importance
            node_sizes.append(node_size)
            
            # Display label with icon
            icon = self.ENTITY_ICONS.get(node_type, '')
            display_text = f"{icon} {node_label}" if icon else node_label
            node_text.append(display_text)
            
            # Hover text with details
            neighbors = list(graph.neighbors(node))
            predecessors = list(graph.predecessors(node))
            
            hover_lines = [
                f"<b>{node_label}</b>",
                f"Type: {node_type.title()}",
                f"Connections: {graph.degree(node)}",
                f"Importance: {centrality:.3f}",
                ""
            ]
            
            if predecessors:
                hover_lines.append(f"Connected from: {', '.join(predecessors[:3])}")
                if len(predecessors) > 3:
                    hover_lines.append(f"  ... and {len(predecessors)-3} more")
            
            if neighbors:
                hover_lines.append(f"Connected to: {', '.join(neighbors[:3])}")
                if len(neighbors) > 3:
                    hover_lines.append(f"  ... and {len(neighbors)-3} more")
            
            node_hover_text.append("<br>".join(hover_lines))
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            text=node_text,
            textposition='top center',
            textfont=dict(size=10, color='#2c3e50', family='Arial Black'),
            hovertext=node_hover_text,
            hoverinfo='text',
            showlegend=False
        )
        
        return node_trace
    
    def _create_figure(self, edge_traces: List, node_trace: go.Scatter, title: str) -> go.Figure:
        """
        Create Plotly figure with all traces
        
        Args:
            edge_traces: List of edge traces
            node_trace: Node trace
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        # Combine all traces
        all_traces = edge_traces + [node_trace]
        
        # Create figure
        fig = go.Figure(
            data=all_traces,
            layout=go.Layout(
                title=dict(
                    text=title,
                    font=dict(size=20, color='#2c3e50'),
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=20, r=20, t=60),
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    title=''
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    title=''
                ),
                plot_bgcolor='#ecf0f1',
                paper_bgcolor='white',
                width=self.width,
                height=self.height,
                annotations=self._create_legend_annotations()
            )
        )
        
        return fig
    
    def _create_legend_annotations(self) -> List[dict]:
        """
        Create manual legend as annotations
        
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        # Legend title
        annotations.append(dict(
            x=1.02,
            y=1.0,
            xref='paper',
            yref='paper',
            text='<b>Entity Types</b>',
            showarrow=False,
            xanchor='left',
            yanchor='top',
            font=dict(size=12)
        ))
        
        # Legend items
        y_pos = 0.95
        for entity_type, color in self.MEDICAL_COLORS.items():
            icon = self.ENTITY_ICONS.get(entity_type, '')
            label = f"{icon} {entity_type.title()}"
            
            # Color box
            annotations.append(dict(
                x=1.02,
                y=y_pos,
                xref='paper',
                yref='paper',
                text='■',
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                font=dict(size=20, color=color)
            ))
            
            # Label
            annotations.append(dict(
                x=1.05,
                y=y_pos,
                xref='paper',
                yref='paper',
                text=label,
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                font=dict(size=10)
            ))
            
            y_pos -= 0.08
        
        return annotations
    
    def _create_empty_plot_html(self, title: str) -> str:
        """Create HTML for empty graph"""
        fig = go.Figure()
        fig.add_annotation(
            text="No data to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color='gray')
        )
        fig.update_layout(
            title=title,
            width=self.width,
            height=self.height,
            plot_bgcolor='#ecf0f1'
        )
        return fig.to_html(include_plotlyjs='cdn', full_html=False)
    
    def _create_error_plot_html(self, error_msg: str) -> str:
        """Create HTML for error case"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization:<br>{error_msg}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color='red')
        )
        fig.update_layout(
            title="Visualization Error",
            width=self.width,
            height=self.height,
            plot_bgcolor='#ecf0f1'
        )
        return fig.to_html(include_plotlyjs='cdn', full_html=False)


# Example usage
if __name__ == "__main__":
    from graph_builder import MedicalGraphBuilder
    
    # Sample triples
    sample_triples = [
        {'subject': 'patient', 'predicate': 'has_condition', 'object': 'hypertension'},
        {'subject': 'patient', 'predicate': 'has_condition', 'object': 'type 2 diabetes'},
        {'subject': 'patient', 'predicate': 'takes', 'object': 'lisinopril 10mg'},
        {'subject': 'patient', 'predicate': 'takes', 'object': 'metformin 500mg'},
        {'subject': 'lisinopril 10mg', 'predicate': 'treats', 'object': 'hypertension'},
        {'subject': 'metformin 500mg', 'predicate': 'treats', 'object': 'type 2 diabetes'},
        {'subject': 'patient', 'predicate': 'underwent', 'object': 'annual physical'},
        {'subject': 'patient', 'predicate': 'received', 'object': 'flu vaccine'}
    ]
    
    # Build graph
    print("Building graph...")
    builder = MedicalGraphBuilder()
    graph = builder.build_graph(sample_triples)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Create visualization
    print("Creating visualization...")
    visualizer = MedicalGraphVisualizer()
    html = visualizer.visualize(graph, title="Sample Medical Knowledge Graph")
    
    # Save to file
    output_file = "medical_kg_visualization.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Medical Knowledge Graph</title>
</head>
<body>
    <h1>Medical Knowledge Graph Visualization</h1>
    {html}
</body>
</html>
""")
    
    print(f"✅ Visualization saved to: {output_file}")
    print("🌐 Open the file in a web browser to view the interactive graph!")
