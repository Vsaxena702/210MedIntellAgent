"""
Medical Knowledge Graph Builder
Constructs NetworkX graphs from SPO triples
"""

import networkx as nx
from typing import List, Dict, Set
import logging

logger = logging.getLogger(__name__)


class MedicalGraphBuilder:
    """Build NetworkX directed graph from medical SPO triples"""
    
    # Medical entity type inference keywords
    ENTITY_TYPE_KEYWORDS = {
        'patient': ['patient', 'individual', 'person'],
        'condition': ['disease', 'disorder', 'syndrome', 'condition', 'diagnosis', 
                     'hypertension', 'diabetes', 'cancer', 'infection', 'inflammation'],
        'medication': ['mg', 'tablet', 'capsule', 'medication', 'drug', 'pill', 
                      'therapy', 'treatment', 'medicine', 'prescription'],
        'procedure': ['surgery', 'operation', 'procedure', 'examination', 'test',
                     'screening', 'biopsy', 'scan', 'imaging'],
        'observation': ['lab', 'result', 'value', 'measurement', 'level', 'count',
                       'blood', 'glucose', 'cholesterol', 'pressure'],
        'immunization': ['vaccine', 'vaccination', 'immunization', 'shot', 
                        'flu shot', 'covid', 'tdap'],
        'encounter': ['visit', 'appointment', 'encounter', 'consultation', 
                     'checkup', 'admission'],
        'provider': ['doctor', 'physician', 'nurse', 'provider', 'practitioner',
                    'specialist', 'surgeon']
    }
    
    # Color scheme for medical entities
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
    
    def build_graph(self, triples: List[Dict[str, str]]) -> nx.DiGraph:
        """
        Build directed graph from SPO triples
        
        Args:
            triples: List of dicts with 'subject', 'predicate', 'object' keys
            
        Returns:
            NetworkX DiGraph with medical metadata
        """
        G = nx.DiGraph()
        
        if not triples:
            logger.warning("No triples provided to build graph")
            return G
        
        # Track all entities for type inference
        all_entities = set()
        for triple in triples:
            all_entities.add(triple['subject'])
            all_entities.add(triple['object'])
        
        # Infer entity types
        entity_types = self._infer_entity_types(all_entities)
        
        # Add nodes and edges
        for triple in triples:
            subject = triple['subject']
            predicate = triple['predicate']
            obj = triple['object']
            
            # Add subject node
            subject_type = entity_types.get(subject, 'unknown')
            G.add_node(
                subject,
                label=self._format_label(subject),
                type=subject_type,
                color=self.MEDICAL_COLORS[subject_type]
            )
            
            # Add object node
            object_type = entity_types.get(obj, 'unknown')
            G.add_node(
                obj,
                label=self._format_label(obj),
                type=object_type,
                color=self.MEDICAL_COLORS[object_type]
            )
            
            # Add edge
            G.add_edge(
                subject, 
                obj,
                label=self._format_label(predicate),
                relationship=predicate
            )
        
        # Calculate centrality for node sizing
        self._calculate_node_metrics(G)
        
        logger.info(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def _infer_entity_types(self, entities: Set[str]) -> Dict[str, str]:
        """
        Infer medical entity types from entity names
        
        Args:
            entities: Set of entity strings
            
        Returns:
            Dict mapping entity -> type
        """
        entity_types = {}
        
        for entity in entities:
            entity_lower = entity.lower()
            
            # Check each type's keywords
            matched_type = 'unknown'
            for entity_type, keywords in self.ENTITY_TYPE_KEYWORDS.items():
                if any(keyword in entity_lower for keyword in keywords):
                    matched_type = entity_type
                    break
            
            entity_types[entity] = matched_type
        
        return entity_types
    
    def _format_label(self, text: str) -> str:
        """
        Format text for display labels
        
        Args:
            text: Raw text
            
        Returns:
            Formatted label (title case, line breaks for long text)
        """
        # Title case
        formatted = text.title()
        
        # Add line breaks for long labels (for better visualization)
        if len(formatted) > 20:
            words = formatted.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) > 20 and current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1
            
            if current_line:
                lines.append(' '.join(current_line))
            
            formatted = '\n'.join(lines)
        
        return formatted
    
    def _calculate_node_metrics(self, graph: nx.DiGraph) -> None:
        """
        Calculate node importance metrics and add as attributes
        
        Args:
            graph: NetworkX graph to update in-place
        """
        if graph.number_of_nodes() == 0:
            return
        
        # Degree centrality (how many connections)
        degree_centrality = nx.degree_centrality(graph)
        nx.set_node_attributes(graph, degree_centrality, 'centrality')
        
        # Calculate node sizes based on centrality
        max_centrality = max(degree_centrality.values()) if degree_centrality else 1
        node_sizes = {}
        for node, centrality in degree_centrality.items():
            # Base size 20, scale up to 70 based on centrality
            size = 20 + (centrality / max_centrality * 50) if max_centrality > 0 else 20
            node_sizes[node] = size
        nx.set_node_attributes(graph, node_sizes, 'size')
        
        # Identify hub nodes (high degree)
        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 0
        hub_nodes = {node for node, degree in degrees.items() 
                     if degree >= max_degree * 0.5}  # Top 50% by degree
        
        is_hub = {node: (node in hub_nodes) for node in graph.nodes()}
        nx.set_node_attributes(graph, is_hub, 'is_hub')
    
    def get_graph_summary(self, graph: nx.DiGraph) -> Dict[str, any]:
        """
        Get summary statistics about the graph
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dict with summary statistics
        """
        if graph.number_of_nodes() == 0:
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'node_types': {},
                'density': 0.0,
                'is_connected': False
            }
        
        # Count nodes by type
        node_types = {}
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Graph metrics
        try:
            density = nx.density(graph)
            is_connected = nx.is_weakly_connected(graph)
        except:
            density = 0.0
            is_connected = False
        
        return {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'node_types': node_types,
            'density': density,
            'is_connected': is_connected
        }


# Example usage
if __name__ == "__main__":
    # Sample triples
    sample_triples = [
        {'subject': 'patient', 'predicate': 'has_condition', 'object': 'hypertension'},
        {'subject': 'patient', 'predicate': 'takes', 'object': 'lisinopril 10mg'},
        {'subject': 'lisinopril 10mg', 'predicate': 'treats', 'object': 'hypertension'},
        {'subject': 'patient', 'predicate': 'has_condition', 'object': 'type 2 diabetes'},
        {'subject': 'patient', 'predicate': 'takes', 'object': 'metformin 500mg'},
        {'subject': 'metformin 500mg', 'predicate': 'treats', 'object': 'type 2 diabetes'},
        {'subject': 'patient', 'predicate': 'underwent', 'object': 'annual physical'},
        {'subject': 'annual physical', 'predicate': 'performed_by', 'object': 'dr. smith'}
    ]
    
    # Build graph
    builder = MedicalGraphBuilder()
    graph = builder.build_graph(sample_triples)
    
    # Print summary
    summary = builder.get_graph_summary(graph)
    print("=== Graph Summary ===")
    print(f"Nodes: {summary['num_nodes']}")
    print(f"Edges: {summary['num_edges']}")
    print(f"Density: {summary['density']:.3f}")
    print(f"Connected: {summary['is_connected']}")
    print(f"\nNode Types:")
    for node_type, count in summary['node_types'].items():
        print(f"  {node_type}: {count}")
    
    # Print sample nodes
    print(f"\n=== Sample Nodes ===")
    for node, data in list(graph.nodes(data=True))[:5]:
        print(f"{node}:")
        print(f"  Type: {data.get('type')}")
        print(f"  Color: {data.get('color')}")
        print(f"  Centrality: {data.get('centrality', 0):.3f}")
        print(f"  Size: {data.get('size', 0):.1f}")
