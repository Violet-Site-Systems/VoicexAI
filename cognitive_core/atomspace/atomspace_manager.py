"""
AtomSpace Manager for EPPN Cognitive Core

Manages the storage, retrieval, and manipulation of atoms in the cognitive system.
Provides the core functionality for maintaining the concept graph and supporting
reasoning operations.
"""

import json
import os
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import threading
from collections import defaultdict

from .atom_types import Atom, AtomType, AttentionValue, TruthValue
from .concept_graph import ConceptGraph


class AtomSpaceManager:
    """
    Manages the AtomSpace for the EPPN cognitive system.
    
    Provides storage, retrieval, and manipulation of atoms with support for:
    - Concept graph representation
    - Attention-based memory management
    - Probabilistic reasoning support
    - Ethical framework integration
    """
    
    def __init__(self, storage_path: str = "cognitive_core/atomspace/storage"):
        """Initialize the AtomSpace manager."""
        self.storage_path = storage_path
        self.atoms: Dict[str, Atom] = {}
        self.concept_graph = ConceptGraph()
        self.lock = threading.RLock()
        
        # Indexes for efficient querying
        self.type_index: Dict[AtomType, Set[str]] = defaultdict(set)
        self.name_index: Dict[str, Set[str]] = defaultdict(set)
        self.content_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Load existing atoms if storage exists
        self._load_atoms()
    
    def add_atom(self, atom: Atom) -> str:
        """Add an atom to the AtomSpace."""
        with self.lock:
            # Update indexes
            self.type_index[atom.atom_type].add(atom.uuid)
            self.name_index[atom.name].add(atom.uuid)
            
            # Index content keywords
            for key, value in atom.content.items():
                if isinstance(value, str):
                    self.content_index[key].add(atom.uuid)
            
            # Store atom
            self.atoms[atom.uuid] = atom
            
            # Update concept graph
            self.concept_graph.add_node(atom.uuid, atom.name, atom.atom_type.value)
            
            # Persist to storage
            self._save_atom(atom)
            
            return atom.uuid
    
    def get_atom(self, uuid: str) -> Optional[Atom]:
        """Retrieve an atom by UUID."""
        with self.lock:
            atom = self.atoms.get(uuid)
            if atom:
                atom.update_access_time()
            return atom
    
    def get_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Get all atoms of a specific type."""
        with self.lock:
            uuids = self.type_index.get(atom_type, set())
            return [self.atoms[uuid] for uuid in uuids if uuid in self.atoms]
    
    def get_atoms_by_name(self, name: str) -> List[Atom]:
        """Get all atoms with a specific name."""
        with self.lock:
            uuids = self.name_index.get(name, set())
            return [self.atoms[uuid] for uuid in uuids if uuid in self.atoms]
    
    def search_atoms(self, query: str, atom_type: Optional[AtomType] = None) -> List[Atom]:
        """Search atoms by content and optionally filter by type."""
        with self.lock:
            results = set()
            
            # Search by name
            for name, uuids in self.name_index.items():
                if query.lower() in name.lower():
                    results.update(uuids)
            
            # Search by content
            for key, uuids in self.content_index.items():
                if query.lower() in key.lower():
                    results.update(uuids)
            
            # Filter by type if specified
            if atom_type:
                type_uuids = self.type_index.get(atom_type, set())
                results = results.intersection(type_uuids)
            
            return [self.atoms[uuid] for uuid in results if uuid in self.atoms]
    
    def create_link(self, source_uuid: str, target_uuid: str, link_type: str = "RELATED"):
        """Create a link between two atoms."""
        with self.lock:
            source_atom = self.atoms.get(source_uuid)
            target_atom = self.atoms.get(target_uuid)
            
            if source_atom and target_atom:
                source_atom.add_outgoing_link(target_uuid)
                target_atom.add_incoming_link(source_uuid)
                
                # Update concept graph
                self.concept_graph.add_edge(source_uuid, target_uuid, link_type)
                
                # Persist changes
                self._save_atom(source_atom)
                self._save_atom(target_atom)
    
    def remove_atom(self, uuid: str) -> bool:
        """Remove an atom from the AtomSpace."""
        with self.lock:
            atom = self.atoms.get(uuid)
            if not atom:
                return False
            
            # Remove from indexes
            self.type_index[atom.atom_type].discard(uuid)
            self.name_index[atom.name].discard(uuid)
            
            for key, value in atom.content.items():
                if isinstance(value, str):
                    self.content_index[key].discard(uuid)
            
            # Remove links
            for linked_uuid in atom.incoming_links + atom.outgoing_links:
                linked_atom = self.atoms.get(linked_uuid)
                if linked_atom:
                    linked_atom.remove_link(uuid)
                    self._save_atom(linked_atom)
            
            # Remove from concept graph
            self.concept_graph.remove_node(uuid)
            
            # Remove from storage
            self._delete_atom_file(uuid)
            del self.atoms[uuid]
            
            return True
    
    def update_attention(self, uuid: str, attention_value: AttentionValue):
        """Update attention value for an atom."""
        with self.lock:
            atom = self.atoms.get(uuid)
            if atom:
                atom.attention_value = attention_value
                atom.updated_at = datetime.now()
                self._save_atom(atom)
    
    def update_truth_value(self, uuid: str, truth_value: TruthValue):
        """Update truth value for an atom."""
        with self.lock:
            atom = self.atoms.get(uuid)
            if atom:
                atom.truth_value = truth_value
                atom.updated_at = datetime.now()
                self._save_atom(atom)
    
    def get_high_attention_atoms(self, threshold: float = 0.5) -> List[Atom]:
        """Get atoms with attention values above threshold."""
        with self.lock:
            high_attention = []
            for atom in self.atoms.values():
                if atom.attention_value.get_total() > threshold:
                    high_attention.append(atom)
            return sorted(high_attention, key=lambda a: a.attention_value.get_total(), reverse=True)
    
    def decay_attention(self, factor: float = 0.9):
        """Apply attention decay to all atoms."""
        with self.lock:
            for atom in self.atoms.values():
                atom.attention_value.decay(factor)
                self._save_atom(atom)
    
    def get_related_atoms(self, uuid: str, max_depth: int = 2) -> List[Atom]:
        """Get atoms related to the given atom within max_depth."""
        with self.lock:
            related_uuids = self.concept_graph.get_related_nodes(uuid, max_depth)
            return [self.atoms[uuid] for uuid in related_uuids if uuid in self.atoms]
    
    def export_atomspace(self, filepath: str):
        """Export the entire AtomSpace to a JSON file."""
        with self.lock:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "atom_count": len(self.atoms),
                "atoms": [atom.to_dict() for atom in self.atoms.values()]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def import_atomspace(self, filepath: str):
        """Import atoms from a JSON file."""
        with self.lock:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            for atom_data in import_data.get("atoms", []):
                atom = Atom.from_dict(atom_data)
                self.add_atom(atom)
    
    def _load_atoms(self):
        """Load atoms from storage directory."""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)
            return
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        atom_data = json.load(f)
                    atom = Atom.from_dict(atom_data)
                    self.atoms[atom.uuid] = atom
                    
                    # Rebuild indexes
                    self.type_index[atom.atom_type].add(atom.uuid)
                    self.name_index[atom.name].add(atom.uuid)
                    for key, value in atom.content.items():
                        if isinstance(value, str):
                            self.content_index[key].add(atom.uuid)
                    
                    # Rebuild concept graph
                    self.concept_graph.add_node(atom.uuid, atom.name, atom.atom_type.value)
                    
                except Exception as e:
                    print(f"Error loading atom from {filepath}: {e}")
    
    def _save_atom(self, atom: Atom):
        """Save an atom to storage."""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)
        
        filepath = os.path.join(self.storage_path, f"{atom.uuid}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(atom.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _delete_atom_file(self, uuid: str):
        """Delete atom file from storage."""
        filepath = os.path.join(self.storage_path, f"{uuid}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get AtomSpace statistics."""
        with self.lock:
            type_counts = {atom_type.value: len(uuids) for atom_type, uuids in self.type_index.items()}
            
            return {
                "total_atoms": len(self.atoms),
                "type_distribution": type_counts,
                "storage_path": self.storage_path,
                "concept_graph_nodes": self.concept_graph.node_count(),
                "concept_graph_edges": self.concept_graph.edge_count()
            }
