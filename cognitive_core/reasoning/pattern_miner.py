"""
Pattern Mining for EPPN Cognitive Core

Implements pattern discovery and mining capabilities for identifying
recurring patterns in policy documents and ethical frameworks.
"""

import re
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import json

from ..atomspace.atom_types import Atom, AtomType
from ..atomspace.atomspace_manager import AtomSpaceManager


@dataclass
class Pattern:
    """Represents a discovered pattern."""
    pattern_id: str
    pattern_type: str
    frequency: int
    confidence: float
    atoms: List[str]
    description: str
    metadata: Dict[str, Any] = None


class PatternMiner:
    """
    Pattern mining system for discovering recurring patterns in policy data.
    
    Capabilities:
    - Text pattern mining
    - Structural pattern discovery
    - Ethical pattern recognition
    - Temporal pattern analysis
    - Concept relationship patterns
    """
    
    def __init__(self, atomspace_manager: AtomSpaceManager):
        """Initialize the pattern miner."""
        self.atomspace = atomspace_manager
        self.discovered_patterns: Dict[str, Pattern] = {}
        
        # Pattern templates for different types
        self.pattern_templates = {
            "text_patterns": [
                r"shall\s+\w+",  # Legal obligations
                r"must\s+\w+",   # Requirements
                r"may\s+\w+",    # Permissions
                r"prohibited\s+from",  # Prohibitions
                r"subject\s+to",  # Conditions
            ],
            "ethical_patterns": [
                r"fair\s+and\s+equitable",
                r"equal\s+opportunity",
                r"non-discriminatory",
                r"reasonable\s+accommodation",
                r"due\s+process",
            ],
            "structural_patterns": [
                r"section\s+\d+",
                r"subsection\s+\([a-z]\)",
                r"paragraph\s+\(\d+\)",
                r"clause\s+\d+",
            ]
        }
        
        # Ethical concept patterns
        self.ethical_concepts = {
            "fairness": ["fair", "equitable", "just", "impartial"],
            "equality": ["equal", "same", "uniform", "consistent"],
            "transparency": ["transparent", "clear", "open", "public"],
            "accountability": ["accountable", "responsible", "liable", "answerable"],
            "privacy": ["private", "confidential", "personal", "sensitive"],
            "autonomy": ["autonomous", "independent", "self-determination", "choice"]
        }
    
    def mine_text_patterns(self, atoms: List[Atom]) -> List[Pattern]:
        """Mine text patterns from policy atoms."""
        patterns = []
        
        for template_name, templates in self.pattern_templates.items():
            for template in templates:
                pattern = self._extract_pattern_from_template(atoms, template, template_name)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def mine_ethical_patterns(self, atoms: List[Atom]) -> List[Pattern]:
        """Mine ethical patterns from policy atoms."""
        ethical_patterns = []
        
        for concept, keywords in self.ethical_concepts.items():
            pattern = self._extract_ethical_pattern(atoms, concept, keywords)
            if pattern:
                ethical_patterns.append(pattern)
        
        return ethical_patterns
    
    def mine_structural_patterns(self, atoms: List[Atom]) -> List[Pattern]:
        """Mine structural patterns from policy atoms."""
        structural_patterns = []
        
        # Analyze document structure
        section_pattern = self._analyze_section_structure(atoms)
        if section_pattern:
            structural_patterns.append(section_pattern)
        
        # Analyze hierarchical relationships
        hierarchy_pattern = self._analyze_hierarchy(atoms)
        if hierarchy_pattern:
            structural_patterns.append(hierarchy_pattern)
        
        return structural_patterns
    
    def mine_concept_relationships(self, atoms: List[Atom]) -> List[Pattern]:
        """Mine patterns in concept relationships."""
        relationship_patterns = []
        
        # Find co-occurrence patterns
        cooccurrence_patterns = self._find_cooccurrence_patterns(atoms)
        relationship_patterns.extend(cooccurrence_patterns)
        
        # Find causal patterns
        causal_patterns = self._find_causal_patterns(atoms)
        relationship_patterns.extend(causal_patterns)
        
        # Find implication patterns
        implication_patterns = self._find_implication_patterns(atoms)
        relationship_patterns.extend(implication_patterns)
        
        return relationship_patterns
    
    def mine_temporal_patterns(self, atoms: List[Atom]) -> List[Pattern]:
        """Mine temporal patterns from atoms with timestamps."""
        temporal_patterns = []
        
        # Group atoms by time periods
        time_groups = self._group_by_time_period(atoms)
        
        # Analyze changes over time
        for period, period_atoms in time_groups.items():
            if len(period_atoms) > 1:
                pattern = self._analyze_temporal_changes(period, period_atoms)
                if pattern:
                    temporal_patterns.append(pattern)
        
        return temporal_patterns
    
    def _extract_pattern_from_template(self, atoms: List[Atom], template: str, pattern_type: str) -> Optional[Pattern]:
        """Extract patterns using regex templates."""
        matches = []
        matching_atoms = []
        
        for atom in atoms:
            content = str(atom.content).lower()
            found_matches = re.findall(template, content, re.IGNORECASE)
            if found_matches:
                matches.extend(found_matches)
                matching_atoms.append(atom.uuid)
        
        if len(matches) >= 2:  # Minimum frequency threshold
            frequency = len(matches)
            confidence = min(1.0, frequency / 10.0)  # Simple confidence calculation
            
            return Pattern(
                pattern_id=f"{pattern_type}_{hash(template)}",
                pattern_type=pattern_type,
                frequency=frequency,
                confidence=confidence,
                atoms=matching_atoms,
                description=f"Pattern: {template} (found {frequency} times)",
                metadata={"template": template, "matches": matches}
            )
        
        return None
    
    def _extract_ethical_pattern(self, atoms: List[Atom], concept: str, keywords: List[str]) -> Optional[Pattern]:
        """Extract ethical concept patterns."""
        matching_atoms = []
        total_occurrences = 0
        
        for atom in atoms:
            content = str(atom.content).lower()
            concept_occurrences = sum(1 for keyword in keywords if keyword in content)
            
            if concept_occurrences > 0:
                matching_atoms.append(atom.uuid)
                total_occurrences += concept_occurrences
        
        if len(matching_atoms) >= 2:
            frequency = len(matching_atoms)
            confidence = min(1.0, total_occurrences / len(keywords))
            
            return Pattern(
                pattern_id=f"ethical_{concept}_{hash(str(keywords))}",
                pattern_type="ethical_concept",
                frequency=frequency,
                confidence=confidence,
                atoms=matching_atoms,
                description=f"Ethical concept '{concept}' pattern (found in {frequency} atoms)",
                metadata={"concept": concept, "keywords": keywords, "total_occurrences": total_occurrences}
            )
        
        return None
    
    def _analyze_section_structure(self, atoms: List[Atom]) -> Optional[Pattern]:
        """Analyze document section structure patterns."""
        section_atoms = [atom for atom in atoms if atom.atom_type == AtomType.POLICY_SECTION]
        
        if len(section_atoms) < 3:
            return None
        
        # Analyze section numbering patterns
        section_numbers = []
        for atom in section_atoms:
            content = str(atom.content)
            numbers = re.findall(r'section\s+(\d+)', content, re.IGNORECASE)
            if numbers:
                section_numbers.extend([int(n) for n in numbers])
        
        if section_numbers:
            section_numbers.sort()
            # Check for sequential numbering
            is_sequential = all(section_numbers[i] == section_numbers[i-1] + 1 
                              for i in range(1, len(section_numbers)))
            
            return Pattern(
                pattern_id="section_structure",
                pattern_type="structural",
                frequency=len(section_numbers),
                confidence=1.0 if is_sequential else 0.5,
                atoms=[atom.uuid for atom in section_atoms],
                description=f"Section structure pattern ({'sequential' if is_sequential else 'non-sequential'} numbering)",
                metadata={"section_numbers": section_numbers, "is_sequential": is_sequential}
            )
        
        return None
    
    def _analyze_hierarchy(self, atoms: List[Atom]) -> Optional[Pattern]:
        """Analyze hierarchical relationships in policy structure."""
        hierarchy_levels = defaultdict(list)
        
        for atom in atoms:
            content = str(atom.content).lower()
            
            # Detect hierarchy indicators
            if "section" in content:
                hierarchy_levels["section"].append(atom.uuid)
            elif "subsection" in content:
                hierarchy_levels["subsection"].append(atom.uuid)
            elif "paragraph" in content:
                hierarchy_levels["paragraph"].append(atom.uuid)
            elif "clause" in content:
                hierarchy_levels["clause"].append(atom.uuid)
        
        if len(hierarchy_levels) >= 2:
            total_atoms = sum(len(atoms) for atoms in hierarchy_levels.values())
            
            return Pattern(
                pattern_id="hierarchy_structure",
                pattern_type="structural",
                frequency=total_atoms,
                confidence=0.8,
                atoms=[uuid for atoms in hierarchy_levels.values() for uuid in atoms],
                description=f"Hierarchical structure with {len(hierarchy_levels)} levels",
                metadata={"hierarchy_levels": dict(hierarchy_levels)}
            )
        
        return None
    
    def _find_cooccurrence_patterns(self, atoms: List[Atom]) -> List[Pattern]:
        """Find co-occurrence patterns between concepts."""
        cooccurrence_patterns = []
        
        # Extract concepts from atoms
        concept_sets = []
        for atom in atoms:
            concepts = self._extract_concepts(atom)
            if concepts:
                concept_sets.append((atom.uuid, concepts))
        
        # Find frequent concept combinations
        concept_combinations = defaultdict(list)
        for atom_uuid, concepts in concept_sets:
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    combination = tuple(sorted([concept1, concept2]))
                    concept_combinations[combination].append(atom_uuid)
        
        # Create patterns for frequent combinations
        for combination, atom_uuids in concept_combinations.items():
            if len(atom_uuids) >= 2:
                pattern = Pattern(
                    pattern_id=f"cooccurrence_{hash(str(combination))}",
                    pattern_type="concept_relationship",
                    frequency=len(atom_uuids),
                    confidence=min(1.0, len(atom_uuids) / 5.0),
                    atoms=atom_uuids,
                    description=f"Co-occurrence of concepts: {combination[0]} and {combination[1]}",
                    metadata={"concepts": combination, "cooccurrence_count": len(atom_uuids)}
                )
                cooccurrence_patterns.append(pattern)
        
        return cooccurrence_patterns
    
    def _find_causal_patterns(self, atoms: List[Atom]) -> List[Pattern]:
        """Find causal relationship patterns."""
        causal_patterns = []
        
        causal_keywords = ["because", "due to", "caused by", "leads to", "results in", "therefore"]
        
        for atom in atoms:
            content = str(atom.content).lower()
            causal_matches = [keyword for keyword in causal_keywords if keyword in content]
            
            if causal_matches:
                pattern = Pattern(
                    pattern_id=f"causal_{hash(atom.uuid)}",
                    pattern_type="concept_relationship",
                    frequency=1,
                    confidence=0.6,
                    atoms=[atom.uuid],
                    description=f"Causal relationship pattern in {atom.name}",
                    metadata={"causal_keywords": causal_matches}
                )
                causal_patterns.append(pattern)
        
        return causal_patterns
    
    def _find_implication_patterns(self, atoms: List[Atom]) -> List[Pattern]:
        """Find implication relationship patterns."""
        implication_patterns = []
        
        implication_keywords = ["if", "then", "implies", "requires", "necessitates"]
        
        for atom in atoms:
            content = str(atom.content).lower()
            implication_matches = [keyword for keyword in implication_keywords if keyword in content]
            
            if implication_matches:
                pattern = Pattern(
                    pattern_id=f"implication_{hash(atom.uuid)}",
                    pattern_type="concept_relationship",
                    frequency=1,
                    confidence=0.7,
                    atoms=[atom.uuid],
                    description=f"Implication pattern in {atom.name}",
                    metadata={"implication_keywords": implication_matches}
                )
                implication_patterns.append(pattern)
        
        return implication_patterns
    
    def _group_by_time_period(self, atoms: List[Atom]) -> Dict[str, List[Atom]]:
        """Group atoms by time periods."""
        time_groups = defaultdict(list)
        
        for atom in atoms:
            # Use creation time for grouping
            time_period = atom.created_at.strftime("%Y-%m")
            time_groups[time_period].append(atom)
        
        return dict(time_groups)
    
    def _analyze_temporal_changes(self, period: str, atoms: List[Atom]) -> Optional[Pattern]:
        """Analyze temporal changes in a time period."""
        if len(atoms) < 2:
            return None
        
        # Simple temporal analysis - count changes in content
        content_changes = 0
        for i in range(1, len(atoms)):
            if atoms[i].content != atoms[i-1].content:
                content_changes += 1
        
        change_rate = content_changes / (len(atoms) - 1) if len(atoms) > 1 else 0
        
        return Pattern(
            pattern_id=f"temporal_{period}",
            pattern_type="temporal",
            frequency=len(atoms),
            confidence=change_rate,
            atoms=[atom.uuid for atom in atoms],
            description=f"Temporal pattern for {period} (change rate: {change_rate:.2f})",
            metadata={"period": period, "change_rate": change_rate, "content_changes": content_changes}
        )
    
    def _extract_concepts(self, atom: Atom) -> List[str]:
        """Extract concepts from an atom's content."""
        content = str(atom.content).lower()
        
        # Simple concept extraction - look for noun phrases
        # This is a simplified version - in practice, you'd use NLP libraries
        words = content.split()
        concepts = []
        
        # Look for capitalized words (potential concepts)
        for word in words:
            if word[0].isupper() and len(word) > 2:
                concepts.append(word.lower())
        
        return concepts
    
    def get_patterns_by_type(self, pattern_type: str) -> List[Pattern]:
        """Get all patterns of a specific type."""
        return [pattern for pattern in self.discovered_patterns.values() 
                if pattern.pattern_type == pattern_type]
    
    def get_high_confidence_patterns(self, threshold: float = 0.7) -> List[Pattern]:
        """Get patterns with confidence above threshold."""
        return [pattern for pattern in self.discovered_patterns.values() 
                if pattern.confidence >= threshold]
    
    def export_patterns(self, filepath: str):
        """Export discovered patterns to a file."""
        patterns_data = {
            "exported_at": self._get_timestamp(),
            "pattern_count": len(self.discovered_patterns),
            "patterns": [
                {
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "frequency": pattern.frequency,
                    "confidence": pattern.confidence,
                    "atoms": pattern.atoms,
                    "description": pattern.description,
                    "metadata": pattern.metadata
                }
                for pattern in self.discovered_patterns.values()
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(patterns_data, f, indent=2, ensure_ascii=False)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()
