"""
Concept Graph for EPPN Cognitive Core

Provides graph-based representation and analysis of concept relationships
in the AtomSpace, supporting reasoning and pattern discovery.
"""

import networkx as nx
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
import json


class ConceptGraph:
    """
    Graph-based representation of concepts and their relationships.
    
    Supports:
    - Concept relationship modeling
    - Path finding and similarity analysis
    - Community detection for concept clustering
    - Centrality analysis for importance ranking
    """
    
    def __init__(self):
        """Initialize the concept graph."""
        self.graph = nx.MultiDiGraph()
        self.node_attributes: Dict[str, Dict[str, Any]] = {}
        self.edge_attributes: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    
    def add_node(self, node_id: str, name: str, node_type: str, attributes: Optional[Dict[str, Any]] = None):
        """Add a node to the concept graph."""
        self.graph.add_node(node_id)
        
        node_attrs = {
            "name": name,
            "type": node_type,
            "created_at": self._get_timestamp()
        }
        
        if attributes:
            node_attrs.update(attributes)
        
        self.node_attributes[node_id] = node_attrs
        self.graph.nodes[node_id].update(node_attrs)
    
    def add_edge(self, source: str, target: str, edge_type: str = "RELATED", attributes: Optional[Dict[str, Any]] = None):
        """Add an edge between two nodes."""
        if source not in self.graph or target not in self.graph:
            return False
        
        edge_key = self.graph.add_edge(source, target, type=edge_type)
        
        edge_attrs = {
            "type": edge_type,
            "created_at": self._get_timestamp()
        }
        
        if attributes:
            edge_attrs.update(attributes)
        
        self.edge_attributes[(source, target, edge_key)] = edge_attrs
        self.graph.edges[source, target, edge_key].update(edge_attrs)
        
        return True
    
    def remove_node(self, node_id: str):
        """Remove a node and all its edges."""
        if node_id in self.graph:
            self.graph.remove_node(node_id)
            self.node_attributes.pop(node_id, None)
            
            # Remove associated edge attributes
            edges_to_remove = [
                key for key in self.edge_attributes.keys()
                if key[0] == node_id or key[1] == node_id
            ]
            for key in edges_to_remove:
                self.edge_attributes.pop(key, None)
    
    def remove_edge(self, source: str, target: str, edge_key: int = 0):
        """Remove an edge between two nodes."""
        if self.graph.has_edge(source, target, edge_key):
            self.graph.remove_edge(source, target, edge_key)
            self.edge_attributes.pop((source, target, edge_key), None)
    
    def get_related_nodes(self, node_id: str, max_depth: int = 2) -> Set[str]:
        """Get nodes related to the given node within max_depth."""
        if node_id not in self.graph:
            return set()
        
        related = set()
        current_level = {node_id}
        
        for depth in range(max_depth):
            next_level = set()
            for node in current_level:
                # Add neighbors
                neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                next_level.update(neighbors)
            
            related.update(next_level)
            current_level = next_level - related  # Avoid revisiting nodes
        
        related.discard(node_id)  # Remove the original node
        return related
    
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find the shortest path between two nodes."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None
    
    def get_node_centrality(self, node_id: str) -> Dict[str, float]:
        """Calculate various centrality measures for a node."""
        if node_id not in self.graph:
            return {}
        
        # Convert to undirected for some centrality measures
        undirected_graph = self.graph.to_undirected()
        
        try:
            betweenness = nx.betweenness_centrality(undirected_graph).get(node_id, 0.0)
            closeness = nx.closeness_centrality(undirected_graph).get(node_id, 0.0)
            degree = nx.degree_centrality(undirected_graph).get(node_id, 0.0)
            
            return {
                "betweenness_centrality": betweenness,
                "closeness_centrality": closeness,
                "degree_centrality": degree
            }
        except:
            return {"betweenness_centrality": 0.0, "closeness_centrality": 0.0, "degree_centrality": 0.0}
    
    def find_communities(self) -> Dict[int, List[str]]:
        """Find communities in the concept graph."""
        try:
            # Convert to undirected for community detection
            undirected_graph = self.graph.to_undirected()
            communities = nx.community.greedy_modularity_communities(undirected_graph)
            
            return {i: list(community) for i, community in enumerate(communities)}
        except:
            return {}
    
    def get_strongly_connected_components(self) -> List[List[str]]:
        """Get strongly connected components."""
        try:
            return list(nx.strongly_connected_components(self.graph))
        except:
            return []
    
    def get_weakly_connected_components(self) -> List[List[str]]:
        """Get weakly connected components."""
        try:
            return list(nx.weakly_connected_components(self.graph))
        except:
            return []
    
    def find_cycles(self) -> List[List[str]]:
        """Find cycles in the graph."""
        try:
            return list(nx.simple_cycles(self.graph))
        except:
            return []
    
    def get_node_neighbors(self, node_id: str) -> Dict[str, List[str]]:
        """Get neighbors of a node categorized by direction."""
        if node_id not in self.graph:
            return {"predecessors": [], "successors": []}
        
        return {
            "predecessors": list(self.graph.predecessors(node_id)),
            "successors": list(self.graph.successors(node_id))
        }
    
    def get_subgraph(self, nodes: List[str]) -> 'ConceptGraph':
        """Create a subgraph containing only the specified nodes."""
        subgraph = ConceptGraph()
        
        # Add nodes
        for node_id in nodes:
            if node_id in self.graph:
                node_attrs = self.node_attributes.get(node_id, {})
                subgraph.add_node(
                    node_id,
                    node_attrs.get("name", ""),
                    node_attrs.get("type", ""),
                    node_attrs
                )
        
        # Add edges between nodes in the subgraph
        for source in nodes:
            for target in nodes:
                if source != target and self.graph.has_edge(source, target):
                    edges = self.graph.get_edge_data(source, target)
                    for edge_key, edge_data in edges.items():
                        subgraph.add_edge(source, target, edge_data.get("type", "RELATED"), edge_data)
        
        return subgraph
    
    def export_graph(self, filepath: str):
        """Export the graph to a file."""
        graph_data = {
            "nodes": [
                {
                    "id": node_id,
                    "attributes": self.node_attributes.get(node_id, {})
                }
                for node_id in self.graph.nodes()
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "key": edge_key,
                    "attributes": self.edge_attributes.get((source, target, edge_key), {})
                }
                for source, target, edge_key in self.graph.edges(keys=True)
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    def import_graph(self, filepath: str):
        """Import a graph from a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # Clear existing graph
        self.graph.clear()
        self.node_attributes.clear()
        self.edge_attributes.clear()
        
        # Add nodes
        for node_data in graph_data.get("nodes", []):
            node_id = node_data["id"]
            attributes = node_data.get("attributes", {})
            
            self.add_node(
                node_id,
                attributes.get("name", ""),
                attributes.get("type", ""),
                attributes
            )
        
        # Add edges
        for edge_data in graph_data.get("edges", []):
            source = edge_data["source"]
            target = edge_data["target"]
            edge_key = edge_data.get("key", 0)
            attributes = edge_data.get("attributes", {})
            
            self.add_edge(source, target, attributes.get("type", "RELATED"), attributes)
    
    def node_count(self) -> int:
        """Get the number of nodes in the graph."""
        return self.graph.number_of_nodes()
    
    def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return self.graph.number_of_edges()
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        try:
            undirected_graph = self.graph.to_undirected()
            
            return {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "average_clustering": nx.average_clustering(undirected_graph),
                "number_of_components": nx.number_weakly_connected_components(self.graph),
                "is_strongly_connected": nx.is_strongly_connected(self.graph),
                "is_weakly_connected": nx.is_weakly_connected(self.graph),
                "has_cycles": len(list(nx.simple_cycles(self.graph))) > 0
            }
        except:
            return {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": 0.0,
                "average_clustering": 0.0,
                "number_of_components": 0,
                "is_strongly_connected": False,
                "is_weakly_connected": False,
                "has_cycles": False
            }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()
