"""
Explainable AI (XAI) Integration for EPPN Cognitive Core

Provides transparency and interpretability for AI decisions in ethical policy analysis.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..atomspace.atom_types import Atom, TruthValue
from .pln_reasoner import PLNReasoner, InferenceResult, InferenceRule


class ExplanationType(Enum):
    """Types of explanations provided by XAI system."""
    
    DECISION_TREE = "DecisionTree"
    FEATURE_IMPORTANCE = "FeatureImportance"
    COUNTERFACTUAL = "Counterfactual"
    ATTENTION_MAP = "AttentionMap"
    RULE_EXTRACTION = "RuleExtraction"
    SENSITIVITY_ANALYSIS = "SensitivityAnalysis"


@dataclass
class Explanation:
    """Structured explanation of an AI decision."""
    
    explanation_type: ExplanationType
    decision_id: str
    confidence: float
    explanation_text: str
    supporting_evidence: List[str]
    alternative_outcomes: List[str] = None
    uncertainty_factors: List[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class DecisionTrace:
    """Trace of decision-making process."""
    
    decision_id: str
    input_atoms: List[str]
    reasoning_steps: List[Dict[str, Any]]
    final_conclusion: str
    confidence_score: float
    explanation: Explanation


class XAIExplainer:
    """
    Explainable AI system for ethical policy analysis.
    
    Provides:
    - Decision traceability and transparency
    - Feature importance analysis
    - Counterfactual explanations
    - Attention visualization
    - Rule extraction from reasoning
    - Sensitivity analysis
    """
    
    def __init__(self, pln_reasoner: PLNReasoner):
        """Initialize the XAI explainer."""
        self.reasoner = pln_reasoner
        self.decision_history: List[DecisionTrace] = []
        self.explanation_cache: Dict[str, Explanation] = {}
        
        # Explanation templates for different scenarios
        self.explanation_templates = {
            "contradiction": "A contradiction was detected between {atom1} and {atom2} because {reason}.",
            "bias_detection": "Potential bias was identified in {context} due to {pattern} patterns.",
            "ethical_implication": "The policy aligns with {framework} because it {reasoning}.",
            "resource_allocation": "Resource allocation analysis shows {finding} based on {criteria}."
        }
    
    def explain_decision(self, decision_id: str, input_atoms: List[Atom], 
                        reasoning_steps: List[Dict[str, Any]], 
                        final_conclusion: str, confidence: float) -> DecisionTrace:
        """Generate a complete explanation for a decision."""
        
        # Create decision trace
        trace = DecisionTrace(
            decision_id=decision_id,
            input_atoms=[atom.uuid for atom in input_atoms],
            reasoning_steps=reasoning_steps,
            final_conclusion=final_conclusion,
            confidence_score=confidence,
            explanation=None
        )
        
        # Generate explanation
        explanation = self._generate_explanation(trace, input_atoms)
        trace.explanation = explanation
        
        # Store in history
        self.decision_history.append(trace)
        
        return trace
    
    def explain_contradiction_detection(self, contradiction_result: InferenceResult) -> Explanation:
        """Explain contradiction detection reasoning."""
        
        premises = [self.reasoner.atomspace.get_atom(uuid) for uuid in contradiction_result.premises]
        premises = [atom for atom in premises if atom is not None]
        
        if len(premises) >= 2:
            atom1, atom2 = premises[0], premises[1]
            
            # Analyze the contradiction
            contradiction_reason = self._analyze_contradiction_reason(atom1, atom2)
            
            explanation_text = self.explanation_templates["contradiction"].format(
                atom1=atom1.name,
                atom2=atom2.name,
                reason=contradiction_reason
            )
            
            return Explanation(
                explanation_type=ExplanationType.RULE_EXTRACTION,
                decision_id=f"contradiction_{atom1.uuid}_{atom2.uuid}",
                confidence=contradiction_result.confidence,
                explanation_text=explanation_text,
                supporting_evidence=contradiction_result.premises,
                alternative_outcomes=["No contradiction detected", "Partial contradiction"],
                uncertainty_factors=["Semantic ambiguity", "Context dependency"],
                metadata={
                    "rule_used": contradiction_result.rule_used.value,
                    "truth_value": contradiction_result.truth_value.__dict__
                }
            )
        
        return None
    
    def explain_bias_detection(self, bias_result: InferenceResult) -> Explanation:
        """Explain bias detection reasoning."""
        
        atom = self.reasoner.atomspace.get_atom(bias_result.premises[0])
        if not atom:
            return None
        
        # Analyze bias context
        bias_context = self._analyze_bias_context(atom, bias_result)
        
        explanation_text = self.explanation_templates["bias_detection"].format(
            context=bias_context["context"],
            pattern=bias_context["pattern"]
        )
        
        return Explanation(
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            decision_id=f"bias_{atom.uuid}",
            confidence=bias_result.confidence,
            explanation_text=explanation_text,
            supporting_evidence=bias_result.premises,
            alternative_outcomes=["No bias detected", "Different bias type"],
            uncertainty_factors=["Subjective interpretation", "Cultural context"],
            metadata={
                "bias_type": bias_context["bias_type"],
                "detected_patterns": bias_context["patterns"]
            }
        )
    
    def explain_ethical_implication(self, ethical_result: InferenceResult) -> Explanation:
        """Explain ethical implication reasoning."""
        
        atom = self.reasoner.atomspace.get_atom(ethical_result.premises[0])
        if not atom:
            return None
        
        # Analyze ethical reasoning
        ethical_analysis = self._analyze_ethical_reasoning(atom, ethical_result)
        
        explanation_text = self.explanation_templates["ethical_implication"].format(
            framework=ethical_analysis["framework"],
            reasoning=ethical_analysis["reasoning"]
        )
        
        return Explanation(
            explanation_type=ExplanationType.DECISION_TREE,
            decision_id=f"ethics_{atom.uuid}",
            confidence=ethical_result.confidence,
            explanation_text=explanation_text,
            supporting_evidence=ethical_result.premises,
            alternative_outcomes=ethical_analysis["alternatives"],
            uncertainty_factors=["Ethical framework selection", "Principle interpretation"],
            metadata={
                "ethical_framework": ethical_analysis["framework"],
                "principles_considered": ethical_analysis["principles"]
            }
        )
    
    def generate_feature_importance(self, atoms: List[Atom]) -> Dict[str, float]:
        """Generate feature importance scores for input atoms."""
        
        importance_scores = {}
        
        for atom in atoms:
            # Calculate importance based on multiple factors
            content_importance = self._calculate_content_importance(atom)
            structural_importance = self._calculate_structural_importance(atom)
            ethical_importance = self._calculate_ethical_importance(atom)
            
            # Weighted combination
            total_importance = (
                0.4 * content_importance +
                0.3 * structural_importance +
                0.3 * ethical_importance
            )
            
            importance_scores[atom.uuid] = total_importance
        
        return importance_scores
    
    def generate_counterfactual_explanation(self, original_atoms: List[Atom], 
                                          target_outcome: str) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations for different outcomes."""
        
        counterfactuals = []
        
        for atom in original_atoms:
            # Generate alternative versions
            alternatives = self._generate_atom_alternatives(atom)
            
            for alt_atom in alternatives:
                # Simulate reasoning with alternative
                alt_result = self._simulate_reasoning([alt_atom])
                
                counterfactual = {
                    "original_atom": atom.uuid,
                    "alternative_atom": alt_atom.uuid,
                    "alternative_content": alt_atom.content,
                    "predicted_outcome": alt_result["outcome"],
                    "confidence_change": alt_result["confidence"] - 0.5,
                    "explanation": f"If '{atom.name}' was '{alt_atom.name}', the outcome would be {alt_result['outcome']}"
                }
                
                counterfactuals.append(counterfactual)
        
        return counterfactuals
    
    def create_attention_visualization(self, atoms: List[Atom], 
                                     attention_weights: Dict[str, float]) -> Dict[str, Any]:
        """Create attention visualization for reasoning process."""
        
        visualization = {
            "atoms": [],
            "attention_weights": attention_weights,
            "reasoning_path": [],
            "focus_areas": []
        }
        
        for atom in atoms:
            atom_info = {
                "uuid": atom.uuid,
                "name": atom.name,
                "content": str(atom.content)[:100] + "..." if len(str(atom.content)) > 100 else str(atom.content),
                "attention_weight": attention_weights.get(atom.uuid, 0.0),
                "importance_rank": self._calculate_importance_rank(atom, attention_weights)
            }
            visualization["atoms"].append(atom_info)
        
        # Sort by attention weight
        visualization["atoms"].sort(key=lambda x: x["attention_weight"], reverse=True)
        
        return visualization
    
    def extract_decision_rules(self, decision_traces: List[DecisionTrace]) -> List[Dict[str, Any]]:
        """Extract decision rules from historical traces."""
        
        rules = []
        
        for trace in decision_traces:
            if trace.explanation:
                rule = {
                    "rule_id": f"rule_{trace.decision_id}",
                    "condition": self._extract_condition(trace),
                    "conclusion": trace.final_conclusion,
                    "confidence": trace.confidence_score,
                    "supporting_evidence": trace.explanation.supporting_evidence,
                    "frequency": self._calculate_rule_frequency(trace, decision_traces)
                }
                rules.append(rule)
        
        return rules
    
    def perform_sensitivity_analysis(self, atoms: List[Atom], 
                                   target_metric: str) -> Dict[str, Any]:
        """Perform sensitivity analysis on input variations."""
        
        sensitivity_results = {
            "metric": target_metric,
            "baseline_score": self._calculate_baseline_score(atoms, target_metric),
            "variations": []
        }
        
        for atom in atoms:
            # Test variations of this atom
            variations = self._generate_atom_variations(atom)
            
            for variation in variations:
                # Create modified atom set
                modified_atoms = [a if a.uuid != atom.uuid else variation for a in atoms]
                
                # Calculate score with variation
                variation_score = self._calculate_baseline_score(modified_atoms, target_metric)
                
                sensitivity_results["variations"].append({
                    "atom_uuid": atom.uuid,
                    "variation_type": variation.name,
                    "score_change": variation_score - sensitivity_results["baseline_score"],
                    "relative_change": (variation_score - sensitivity_results["baseline_score"]) / sensitivity_results["baseline_score"] if sensitivity_results["baseline_score"] != 0 else 0
                })
        
        return sensitivity_results
    
    def _generate_explanation(self, trace: DecisionTrace, atoms: List[Atom]) -> Explanation:
        """Generate explanation for a decision trace."""
        
        # Determine explanation type based on reasoning steps
        if any(step.get("type") == "contradiction" for step in trace.reasoning_steps):
            explanation_type = ExplanationType.RULE_EXTRACTION
        elif any(step.get("type") == "bias" for step in trace.reasoning_steps):
            explanation_type = ExplanationType.FEATURE_IMPORTANCE
        else:
            explanation_type = ExplanationType.DECISION_TREE
        
        # Generate explanation text
        explanation_text = self._create_explanation_text(trace, atoms)
        
        # Extract supporting evidence
        supporting_evidence = [step.get("evidence", []) for step in trace.reasoning_steps]
        supporting_evidence = [item for sublist in supporting_evidence for item in sublist]
        
        return Explanation(
            explanation_type=explanation_type,
            decision_id=trace.decision_id,
            confidence=trace.confidence_score,
            explanation_text=explanation_text,
            supporting_evidence=supporting_evidence,
            alternative_outcomes=self._generate_alternative_outcomes(trace),
            uncertainty_factors=self._identify_uncertainty_factors(trace),
            metadata={"reasoning_steps": len(trace.reasoning_steps)}
        )
    
    def _analyze_contradiction_reason(self, atom1: Atom, atom2: Atom) -> str:
        """Analyze the reason for contradiction between two atoms."""
        
        content1 = str(atom1.content).lower()
        content2 = str(atom2.content).lower()
        
        # Check for specific contradiction patterns
        if "allow" in content1 and "prohibit" in content2:
            return "one statement allows while the other prohibits the same action"
        elif "require" in content1 and "forbid" in content2:
            return "one statement requires while the other forbids the same action"
        elif "increase" in content1 and "decrease" in content2:
            return "one statement increases while the other decreases the same metric"
        else:
            return "conflicting semantic meanings detected in the statements"
    
    def _analyze_bias_context(self, atom: Atom, bias_result: InferenceResult) -> Dict[str, Any]:
        """Analyze the context of bias detection."""
        
        content = str(atom.content).lower()
        
        # Determine bias type and context
        bias_type = "unknown"
        context = "policy content"
        pattern = "language patterns"
        
        if "discrimination" in bias_result.conclusion:
            bias_type = "discrimination"
            context = "treatment of different groups"
        elif "confirmation" in bias_result.conclusion:
            bias_type = "confirmation_bias"
            context = "evidence selection"
        elif "anchoring" in bias_result.conclusion:
            bias_type = "anchoring"
            context = "reference point selection"
        
        return {
            "bias_type": bias_type,
            "context": context,
            "pattern": pattern,
            "patterns": [bias_result.conclusion]
        }
    
    def _analyze_ethical_reasoning(self, atom: Atom, ethical_result: InferenceResult) -> Dict[str, Any]:
        """Analyze ethical reasoning for an atom."""
        
        content = str(atom.content).lower()
        
        # Determine framework and reasoning
        framework = "general_ethics"
        reasoning = "general ethical principles"
        alternatives = ["No ethical implications", "Different ethical framework"]
        principles = ["fairness", "justice"]
        
        if "utilitarian" in ethical_result.conclusion:
            framework = "utilitarian"
            reasoning = "maximizes overall benefit"
        elif "deontological" in ethical_result.conclusion:
            framework = "deontological"
            reasoning = "follows moral duties"
        elif "virtue" in ethical_result.conclusion:
            framework = "virtue_ethics"
            reasoning = "promotes virtuous character"
        
        return {
            "framework": framework,
            "reasoning": reasoning,
            "alternatives": alternatives,
            "principles": principles
        }
    
    def _calculate_content_importance(self, atom: Atom) -> float:
        """Calculate content-based importance score."""
        
        content = str(atom.content)
        
        # Factors that increase importance
        importance_factors = {
            "length": min(1.0, len(content) / 500),  # Longer content is more important
            "keywords": self._count_important_keywords(content),
            "structure": self._analyze_content_structure(content)
        }
        
        return sum(importance_factors.values()) / len(importance_factors)
    
    def _calculate_structural_importance(self, atom: Atom) -> float:
        """Calculate structural importance score."""
        
        # Based on atom type and relationships
        type_importance = {
            "policy": 0.9,
            "principle": 0.8,
            "evidence": 0.7,
            "context": 0.6
        }
        
        return type_importance.get(atom.atom_type.value, 0.5)
    
    def _calculate_ethical_importance(self, atom: Atom) -> float:
        """Calculate ethical importance score."""
        
        content = str(atom.content).lower()
        
        # Ethical keywords that increase importance
        ethical_keywords = [
            "fair", "just", "equal", "right", "wrong", "harm", "benefit",
            "discrimination", "bias", "equity", "justice", "ethics"
        ]
        
        keyword_count = sum(1 for keyword in ethical_keywords if keyword in content)
        return min(1.0, keyword_count / len(ethical_keywords))
    
    def _count_important_keywords(self, content: str) -> float:
        """Count important keywords in content."""
        
        important_keywords = [
            "policy", "regulation", "law", "rule", "guideline", "standard",
            "requirement", "mandatory", "prohibited", "allowed", "permitted"
        ]
        
        content_lower = content.lower()
        count = sum(1 for keyword in important_keywords if keyword in content_lower)
        return min(1.0, count / len(important_keywords))
    
    def _analyze_content_structure(self, content: str) -> float:
        """Analyze content structure for importance."""
        
        # Check for structured elements
        structure_indicators = [
            "section", "subsection", "article", "clause", "paragraph",
            "numbered", "bulleted", "list", "table", "figure"
        ]
        
        content_lower = content.lower()
        count = sum(1 for indicator in structure_indicators if indicator in content_lower)
        return min(1.0, count / len(structure_indicators))
    
    def _generate_atom_alternatives(self, atom: Atom) -> List[Atom]:
        """Generate alternative versions of an atom."""
        
        alternatives = []
        content = str(atom.content)
        
        # Generate semantic alternatives
        if "allow" in content.lower():
            alt_content = content.replace("allow", "prohibit")
            alt_atom = Atom(
                name=f"{atom.name}_alternative",
                content=alt_content,
                atom_type=atom.atom_type,
                truth_value=TruthValue(strength=0.3, confidence=0.5)
            )
            alternatives.append(alt_atom)
        
        if "require" in content.lower():
            alt_content = content.replace("require", "recommend")
            alt_atom = Atom(
                name=f"{atom.name}_alternative",
                content=alt_content,
                atom_type=atom.atom_type,
                truth_value=TruthValue(strength=0.6, confidence=0.7)
            )
            alternatives.append(alt_atom)
        
        return alternatives
    
    def _simulate_reasoning(self, atoms: List[Atom]) -> Dict[str, Any]:
        """Simulate reasoning with alternative atoms."""
        
        # Simple simulation - in practice, this would use the actual reasoning engine
        return {
            "outcome": "alternative_outcome",
            "confidence": 0.6
        }
    
    def _calculate_importance_rank(self, atom: Atom, attention_weights: Dict[str, float]) -> int:
        """Calculate importance rank for an atom."""
        
        weight = attention_weights.get(atom.uuid, 0.0)
        
        if weight > 0.8:
            return 1  # High importance
        elif weight > 0.6:
            return 2  # Medium-high importance
        elif weight > 0.4:
            return 3  # Medium importance
        elif weight > 0.2:
            return 4  # Low-medium importance
        else:
            return 5  # Low importance
    
    def _extract_condition(self, trace: DecisionTrace) -> str:
        """Extract condition from decision trace."""
        
        if trace.reasoning_steps:
            first_step = trace.reasoning_steps[0]
            return first_step.get("condition", "unknown_condition")
        
        return "no_condition"
    
    def _calculate_rule_frequency(self, trace: DecisionTrace, all_traces: List[DecisionTrace]) -> int:
        """Calculate frequency of a rule across all traces."""
        
        # Count similar traces
        similar_count = 0
        for other_trace in all_traces:
            if (other_trace.final_conclusion == trace.final_conclusion and
                other_trace.decision_id != trace.decision_id):
                similar_count += 1
        
        return similar_count
    
    def _calculate_baseline_score(self, atoms: List[Atom], metric: str) -> float:
        """Calculate baseline score for a metric."""
        
        # Simple baseline calculation
        if metric == "ethical_score":
            return sum(self.reasoner._calculate_ethical_score(atom) for atom in atoms) / len(atoms)
        elif metric == "contradiction_score":
            contradictions = self.reasoner.detect_contradictions(atoms)
            return len(contradictions) / len(atoms) if atoms else 0
        else:
            return 0.5  # Default neutral score
    
    def _generate_atom_variations(self, atom: Atom) -> List[Atom]:
        """Generate variations of an atom for sensitivity analysis."""
        
        variations = []
        content = str(atom.content)
        
        # Generate variations by modifying content
        variations.append(Atom(
            name=f"{atom.name}_stronger",
            content=content.replace("may", "must").replace("should", "shall"),
            atom_type=atom.atom_type,
            truth_value=TruthValue(strength=0.9, confidence=0.8)
        ))
        
        variations.append(Atom(
            name=f"{atom.name}_weaker",
            content=content.replace("must", "may").replace("shall", "should"),
            atom_type=atom.atom_type,
            truth_value=TruthValue(strength=0.3, confidence=0.6)
        ))
        
        return variations
    
    def _create_explanation_text(self, trace: DecisionTrace, atoms: List[Atom]) -> str:
        """Create human-readable explanation text."""
        
        explanation_parts = [
            f"Decision: {trace.final_conclusion}",
            f"Confidence: {trace.confidence_score:.2f}",
            f"Based on {len(atoms)} input elements and {len(trace.reasoning_steps)} reasoning steps."
        ]
        
        return " ".join(explanation_parts)
    
    def _generate_alternative_outcomes(self, trace: DecisionTrace) -> List[str]:
        """Generate alternative outcomes for a decision."""
        
        return [
            "Different conclusion with same evidence",
            "Same conclusion with different reasoning",
            "No conclusion due to insufficient evidence"
        ]
    
    def _identify_uncertainty_factors(self, trace: DecisionTrace) -> List[str]:
        """Identify factors contributing to uncertainty."""
        
        uncertainty_factors = []
        
        if trace.confidence_score < 0.7:
            uncertainty_factors.append("Low confidence in reasoning")
        
        if len(trace.reasoning_steps) < 3:
            uncertainty_factors.append("Limited reasoning steps")
        
        if not trace.explanation or not trace.explanation.supporting_evidence:
            uncertainty_factors.append("Limited supporting evidence")
        
        return uncertainty_factors
    
    def get_explanation_summary(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of explanations for a decision."""
        
        trace = next((t for t in self.decision_history if t.decision_id == decision_id), None)
        
        if not trace or not trace.explanation:
            return None
        
        return {
            "decision_id": decision_id,
            "conclusion": trace.final_conclusion,
            "confidence": trace.confidence_score,
            "explanation_type": trace.explanation.explanation_type.value,
            "explanation_text": trace.explanation.explanation_text,
            "supporting_evidence_count": len(trace.explanation.supporting_evidence),
            "uncertainty_factors": trace.explanation.uncertainty_factors
        }
    
    def export_explanations(self, format: str = "json") -> Dict[str, Any]:
        """Export all explanations in specified format."""
        
        export_data = {
            "total_decisions": len(self.decision_history),
            "explanations": []
        }
        
        for trace in self.decision_history:
            if trace.explanation:
                explanation_data = {
                    "decision_id": trace.decision_id,
                    "conclusion": trace.final_conclusion,
                    "confidence": trace.confidence_score,
                    "explanation": trace.explanation.__dict__
                }
                export_data["explanations"].append(explanation_data)
        
        return export_data
