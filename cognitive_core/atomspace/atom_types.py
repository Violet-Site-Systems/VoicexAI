"""
Atom types and data structures for the EPPN cognitive core.

Defines the fundamental building blocks for representing policy concepts,
ethical frameworks, and reasoning structures in the AtomSpace.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import json
import uuid
from datetime import datetime


class AtomType(Enum):
    """Core atom types for policy and ethical reasoning."""
    
    # Policy-related atoms
    POLICY_DOCUMENT = "PolicyDocument"
    POLICY_SECTION = "PolicySection"
    POLICY_CLAUSE = "PolicyClause"
    POLICY_CONCEPT = "PolicyConcept"
    
    # Ethical framework atoms
    ETHICAL_PRINCIPLE = "EthicalPrinciple"
    ETHICAL_VALUE = "EthicalValue"
    ETHICAL_CONSTRAINT = "EthicalConstraint"
    ETHICAL_VIOLATION = "EthicalViolation"
    
    # Reasoning atoms
    LOGICAL_RULE = "LogicalRule"
    INFERENCE_RULE = "InferenceRule"
    PROBABILITY_DISTRIBUTION = "ProbabilityDistribution"
    EVIDENCE = "Evidence"
    
    # Relationship atoms
    IMPLICATION = "Implication"
    CONTRADICTION = "Contradiction"
    SIMILARITY = "Similarity"
    CAUSALITY = "Causality"
    
    # Attention and importance
    ATTENTION_VALUE = "AttentionValue"
    IMPORTANCE = "Importance"
    RELEVANCE = "Relevance"


@dataclass
class AttentionValue:
    """ECAN-style attention values for atoms."""
    short_term: float = 0.0
    long_term: float = 0.0
    very_long_term: float = 0.0
    
    def get_total(self) -> float:
        """Calculate total attention value."""
        return self.short_term + self.long_term + self.very_long_term
    
    def decay(self, factor: float = 0.9):
        """Apply attention decay."""
        self.short_term *= factor
        self.long_term *= factor
        self.very_long_term *= factor


@dataclass
class TruthValue:
    """Probabilistic truth values for atoms."""
    strength: float = 1.0  # Confidence in the truth
    confidence: float = 1.0  # Confidence in the strength estimate
    
    def __post_init__(self):
        """Validate truth value bounds."""
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class Atom:
    """Core atom data structure for the cognitive system."""
    
    # Core identification
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    atom_type: AtomType = AtomType.POLICY_CONCEPT
    
    # Content and meaning
    name: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Cognitive properties
    attention_value: AttentionValue = field(default_factory=AttentionValue)
    truth_value: TruthValue = field(default_factory=TruthValue)
    
    # Relationships
    incoming_links: List[str] = field(default_factory=list)
    outgoing_links: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert atom to dictionary for serialization."""
        return {
            "uuid": self.uuid,
            "atom_type": self.atom_type.value,
            "name": self.name,
            "content": self.content,
            "metadata": self.metadata,
            "attention_value": {
                "short_term": self.attention_value.short_term,
                "long_term": self.attention_value.long_term,
                "very_long_term": self.attention_value.very_long_term
            },
            "truth_value": {
                "strength": self.truth_value.strength,
                "confidence": self.truth_value.confidence
            },
            "incoming_links": self.incoming_links,
            "outgoing_links": self.outgoing_links,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Atom':
        """Create atom from dictionary."""
        atom = cls()
        atom.uuid = data["uuid"]
        atom.atom_type = AtomType(data["atom_type"])
        atom.name = data["name"]
        atom.content = data["content"]
        atom.metadata = data["metadata"]
        
        # Restore attention value
        av_data = data["attention_value"]
        atom.attention_value = AttentionValue(
            short_term=av_data["short_term"],
            long_term=av_data["long_term"],
            very_long_term=av_data["very_long_term"]
        )
        
        # Restore truth value
        tv_data = data["truth_value"]
        atom.truth_value = TruthValue(
            strength=tv_data["strength"],
            confidence=tv_data["confidence"]
        )
        
        atom.incoming_links = data["incoming_links"]
        atom.outgoing_links = data["outgoing_links"]
        atom.created_at = datetime.fromisoformat(data["created_at"])
        atom.updated_at = datetime.fromisoformat(data["updated_at"])
        atom.last_accessed = datetime.fromisoformat(data["last_accessed"])
        
        return atom
    
    def update_access_time(self):
        """Update last accessed timestamp."""
        self.last_accessed = datetime.now()
    
    def add_incoming_link(self, atom_uuid: str):
        """Add incoming link to another atom."""
        if atom_uuid not in self.incoming_links:
            self.incoming_links.append(atom_uuid)
            self.updated_at = datetime.now()
    
    def add_outgoing_link(self, atom_uuid: str):
        """Add outgoing link to another atom."""
        if atom_uuid not in self.outgoing_links:
            self.outgoing_links.append(atom_uuid)
            self.updated_at = datetime.now()
    
    def remove_link(self, atom_uuid: str):
        """Remove link to another atom."""
        if atom_uuid in self.incoming_links:
            self.incoming_links.remove(atom_uuid)
        if atom_uuid in self.outgoing_links:
            self.outgoing_links.remove(atom_uuid)
        self.updated_at = datetime.now()


class PolicyAtom(Atom):
    """Specialized atom for policy-related concepts."""
    
    def __init__(self, name: str, policy_content: Dict[str, Any], **kwargs):
        super().__init__(
            atom_type=AtomType.POLICY_CONCEPT,
            name=name,
            content=policy_content,
            **kwargs
        )


class EthicalAtom(Atom):
    """Specialized atom for ethical concepts and principles."""
    
    def __init__(self, name: str, ethical_framework: str, principle: str, **kwargs):
        super().__init__(
            atom_type=AtomType.ETHICAL_PRINCIPLE,
            name=name,
            content={
                "framework": ethical_framework,
                "principle": principle
            },
            **kwargs
        )


class ReasoningAtom(Atom):
    """Specialized atom for logical reasoning and inference."""
    
    def __init__(self, name: str, rule_type: str, premises: List[str], conclusion: str, **kwargs):
        super().__init__(
            atom_type=AtomType.LOGICAL_RULE,
            name=name,
            content={
                "rule_type": rule_type,
                "premises": premises,
                "conclusion": conclusion
            },
            **kwargs
        )
