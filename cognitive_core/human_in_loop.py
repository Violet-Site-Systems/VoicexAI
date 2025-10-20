"""
Human-in-the-Loop Integration for EPPN

Provides interfaces for human oversight, feedback, and decision-making in the ethical policy analysis pipeline.
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

from .atomspace.atom_types import Atom
from .reasoning.ethical_analyzer import EthicalAnalyzer
from .reasoning.xai_explainer import XAIExplainer, DecisionTrace


class HumanInteractionType(Enum):
    """Types of human interactions in the loop."""
    
    APPROVAL = "approval"
    FEEDBACK = "feedback"
    CLARIFICATION = "clarification"
    OVERRIDE = "override"
    ESCALATION = "escalation"


class InteractionStatus(Enum):
    """Status of human interactions."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class HumanInteraction:
    """Represents a human interaction request."""
    
    interaction_id: str
    interaction_type: HumanInteractionType
    title: str
    description: str
    context: Dict[str, Any]
    required_action: str
    options: List[Dict[str, Any]] = None
    deadline: Optional[datetime] = None
    priority: int = 1  # 1-5, 5 being highest
    status: InteractionStatus = InteractionStatus.PENDING
    created_at: datetime = None
    human_response: Optional[Dict[str, Any]] = None
    response_timestamp: Optional[datetime] = None
    dashboard_url: Optional[str] = None
    chat_url: Optional[str] = None


@dataclass
class HumanFeedback:
    """Represents human feedback on AI decisions."""
    
    feedback_id: str
    decision_id: str
    feedback_type: str
    rating: int  # 1-5 scale
    comments: str
    suggested_improvements: List[str] = None
    timestamp: datetime = None
    human_id: Optional[str] = None


class HumanInTheLoop:
    """
    Human-in-the-loop integration system for ethical policy analysis.
    
    Provides:
    - Human approval workflows
    - Feedback collection and processing
    - Escalation mechanisms
    - Dashboard and chat integration
    - Decision override capabilities
    """
    
    def __init__(self, ethical_analyzer: EthicalAnalyzer, xai_explainer: XAIExplainer,
                 base_url: str = "http://localhost:8000"):
        """Initialize the human-in-the-loop system."""
        self.ethical_analyzer = ethical_analyzer
        self.xai_explainer = xai_explainer
        self.base_url = base_url
        
        # Storage for interactions and feedback
        self.pending_interactions: Dict[str, HumanInteraction] = {}
        self.completed_interactions: Dict[str, HumanInteraction] = {}
        self.feedback_history: List[HumanFeedback] = []
        
        # Callbacks for different interaction types
        self.interaction_callbacks: Dict[HumanInteractionType, Callable] = {}
        
        # Configuration
        self.auto_escalation_timeout = 24 * 60 * 60  # 24 hours in seconds
        self.require_human_approval_threshold = 0.3  # Below this confidence, require approval
    
    def request_human_approval(self, decision_trace: DecisionTrace, 
                             context: Dict[str, Any] = None) -> HumanInteraction:
        """Request human approval for a decision."""
        
        interaction_id = str(uuid.uuid4())
        
        # Generate explanation for human review
        explanation = self.xai_explainer.explain_decision(
            decision_trace.decision_id,
            [self.ethical_analyzer.atomspace.get_atom(uuid) for uuid in decision_trace.input_atoms],
            decision_trace.reasoning_steps,
            decision_trace.final_conclusion,
            decision_trace.confidence_score
        )
        
        # Create interaction request
        interaction = HumanInteraction(
            interaction_id=interaction_id,
            interaction_type=HumanInteractionType.APPROVAL,
            title=f"Approval Required: {decision_trace.final_conclusion[:50]}...",
            description=f"AI decision requires human approval due to confidence level: {decision_trace.confidence_score:.2f}",
            context={
                "decision_trace": asdict(decision_trace),
                "explanation": asdict(explanation.explanation) if explanation else None,
                "additional_context": context or {}
            },
            required_action="Review and approve or reject the AI decision",
            options=[
                {"action": "approve", "label": "Approve Decision", "description": "Accept the AI decision"},
                {"action": "reject", "label": "Reject Decision", "description": "Reject the AI decision"},
                {"action": "modify", "label": "Request Modification", "description": "Request changes to the decision"},
                {"action": "escalate", "label": "Escalate", "description": "Escalate to higher authority"}
            ],
            priority=self._calculate_priority(decision_trace),
            dashboard_url=f"{self.base_url}/dashboard/approval/{interaction_id}",
            chat_url=f"{self.base_url}/chat/approval/{interaction_id}",
            created_at=datetime.now()
        )
        
        self.pending_interactions[interaction_id] = interaction
        
        # Send notification
        self._send_notification(interaction)
        
        return interaction
    
    def request_human_feedback(self, decision_trace: DecisionTrace, 
                              feedback_type: str = "general") -> HumanInteraction:
        """Request human feedback on a decision."""
        
        interaction_id = str(uuid.uuid4())
        
        interaction = HumanInteraction(
            interaction_id=interaction_id,
            interaction_type=HumanInteractionType.FEEDBACK,
            title=f"Feedback Request: {decision_trace.final_conclusion[:50]}...",
            description="Please provide feedback on the AI decision to improve future performance",
            context={
                "decision_trace": asdict(decision_trace),
                "feedback_type": feedback_type
            },
            required_action="Provide feedback on the AI decision quality and accuracy",
            options=[
                {"action": "rate", "label": "Rate Decision", "description": "Rate the decision quality (1-5)"},
                {"action": "comment", "label": "Add Comments", "description": "Provide detailed comments"},
                {"action": "suggest", "label": "Suggest Improvements", "description": "Suggest improvements"}
            ],
            priority=2,
            dashboard_url=f"{self.base_url}/dashboard/feedback/{interaction_id}",
            chat_url=f"{self.base_url}/chat/feedback/{interaction_id}",
            created_at=datetime.now()
        )
        
        self.pending_interactions[interaction_id] = interaction
        
        return interaction
    
    def request_clarification(self, question: str, context: Dict[str, Any] = None) -> HumanInteraction:
        """Request clarification from human expert."""
        
        interaction_id = str(uuid.uuid4())
        
        interaction = HumanInteraction(
            interaction_id=interaction_id,
            interaction_type=HumanInteractionType.CLARIFICATION,
            title=f"Clarification Needed: {question[:50]}...",
            description=question,
            context=context or {},
            required_action="Provide clarification or additional information",
            priority=3,
            dashboard_url=f"{self.base_url}/dashboard/clarification/{interaction_id}",
            chat_url=f"{self.base_url}/chat/clarification/{interaction_id}",
            created_at=datetime.now()
        )
        
        self.pending_interactions[interaction_id] = interaction
        
        return interaction
    
    def escalate_decision(self, decision_trace: DecisionTrace, 
                         escalation_reason: str) -> HumanInteraction:
        """Escalate a decision to human authority."""
        
        interaction_id = str(uuid.uuid4())
        
        interaction = HumanInteraction(
            interaction_id=interaction_id,
            interaction_type=HumanInteractionType.ESCALATION,
            title=f"Escalation Required: {decision_trace.final_conclusion[:50]}...",
            description=f"Decision escalated due to: {escalation_reason}",
            context={
                "decision_trace": asdict(decision_trace),
                "escalation_reason": escalation_reason
            },
            required_action="Review escalated decision and provide final determination",
            priority=5,  # Highest priority
            dashboard_url=f"{self.base_url}/dashboard/escalation/{interaction_id}",
            chat_url=f"{self.base_url}/chat/escalation/{interaction_id}",
            created_at=datetime.now()
        )
        
        self.pending_interactions[interaction_id] = interaction
        
        return interaction
    
    def process_human_response(self, interaction_id: str, 
                              response: Dict[str, Any]) -> bool:
        """Process human response to an interaction."""
        
        if interaction_id not in self.pending_interactions:
            return False
        
        interaction = self.pending_interactions[interaction_id]
        interaction.human_response = response
        interaction.response_timestamp = datetime.now()
        interaction.status = InteractionStatus.COMPLETED
        
        # Move to completed interactions
        self.completed_interactions[interaction_id] = interaction
        del self.pending_interactions[interaction_id]
        
        # Process the response based on interaction type
        success = self._process_response_by_type(interaction, response)
        
        # Store feedback if applicable
        if interaction.interaction_type == HumanInteractionType.FEEDBACK:
            self._store_feedback(interaction, response)
        
        return success
    
    def should_require_human_approval(self, decision_trace: DecisionTrace) -> bool:
        """Determine if human approval is required for a decision."""
        
        # Check confidence threshold
        if decision_trace.confidence_score < self.require_human_approval_threshold:
            return True
        
        # Check for high-risk decisions
        if self._is_high_risk_decision(decision_trace):
            return True
        
        # Check for contradictory evidence
        if self._has_contradictory_evidence(decision_trace):
            return True
        
        return False
    
    def get_pending_interactions(self, human_id: str = None) -> List[HumanInteraction]:
        """Get pending interactions for a human."""
        
        # Filter by human_id if provided (for multi-user scenarios)
        pending = list(self.pending_interactions.values())
        
        if human_id:
            # In a real implementation, you'd filter by assigned human
            pass
        
        # Sort by priority and creation time
        pending.sort(key=lambda x: (-x.priority, x.created_at))
        
        return pending
    
    def get_interaction_status(self, interaction_id: str) -> Optional[InteractionStatus]:
        """Get status of a specific interaction."""
        
        if interaction_id in self.pending_interactions:
            return self.pending_interactions[interaction_id].status
        elif interaction_id in self.completed_interactions:
            return self.completed_interactions[interaction_id].status
        
        return None
    
    def send_text_notification(self, interaction: HumanInteraction, 
                              phone_number: str = None, 
                              email: str = None) -> bool:
        """Send text notification about interaction."""
        
        message = self._format_notification_message(interaction)
        
        # In a real implementation, you'd integrate with SMS/email services
        # For now, we'll simulate the notification
        
        notification_data = {
            "interaction_id": interaction.interaction_id,
            "message": message,
            "dashboard_url": interaction.dashboard_url,
            "chat_url": interaction.chat_url,
            "phone_number": phone_number,
            "email": email,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store notification for tracking
        self._log_notification(notification_data)
        
        return True
    
    def generate_dashboard_link(self, interaction: HumanInteraction) -> str:
        """Generate dashboard link for interaction."""
        
        return f"{self.base_url}/dashboard/{interaction.interaction_type.value}/{interaction.interaction_id}"
    
    def generate_chat_link(self, interaction: HumanInteraction) -> str:
        """Generate chat link for interaction."""
        
        return f"{self.base_url}/chat/{interaction.interaction_type.value}/{interaction.interaction_id}"
    
    def _calculate_priority(self, decision_trace: DecisionTrace) -> int:
        """Calculate priority for an interaction based on decision characteristics."""
        
        priority = 1  # Default priority
        
        # Higher priority for lower confidence
        if decision_trace.confidence_score < 0.3:
            priority = 5
        elif decision_trace.confidence_score < 0.5:
            priority = 4
        elif decision_trace.confidence_score < 0.7:
            priority = 3
        elif decision_trace.confidence_score < 0.8:
            priority = 2
        
        # Check for high-risk indicators
        if self._is_high_risk_decision(decision_trace):
            priority = max(priority, 4)
        
        return priority
    
    def _is_high_risk_decision(self, decision_trace: DecisionTrace) -> bool:
        """Determine if a decision is high-risk."""
        
        conclusion = decision_trace.final_conclusion.lower()
        
        # High-risk keywords
        high_risk_keywords = [
            "discrimination", "bias", "unfair", "harm", "violation",
            "illegal", "unethical", "contradiction", "conflict"
        ]
        
        return any(keyword in conclusion for keyword in high_risk_keywords)
    
    def _has_contradictory_evidence(self, decision_trace: DecisionTrace) -> bool:
        """Check if decision has contradictory evidence."""
        
        # Look for contradiction indicators in reasoning steps
        for step in decision_trace.reasoning_steps:
            if step.get("type") == "contradiction":
                return True
        
        return False
    
    def _process_response_by_type(self, interaction: HumanInteraction, 
                                 response: Dict[str, Any]) -> bool:
        """Process response based on interaction type."""
        
        if interaction.interaction_type == HumanInteractionType.APPROVAL:
            return self._process_approval_response(interaction, response)
        elif interaction.interaction_type == HumanInteractionType.FEEDBACK:
            return self._process_feedback_response(interaction, response)
        elif interaction.interaction_type == HumanInteractionType.CLARIFICATION:
            return self._process_clarification_response(interaction, response)
        elif interaction.interaction_type == HumanInteractionType.ESCALATION:
            return self._process_escalation_response(interaction, response)
        
        return False
    
    def _process_approval_response(self, interaction: HumanInteraction, 
                                  response: Dict[str, Any]) -> bool:
        """Process approval response."""
        
        action = response.get("action")
        
        if action == "approve":
            # Decision approved, continue with normal flow
            return True
        elif action == "reject":
            # Decision rejected, may need to retry or use alternative approach
            return False
        elif action == "modify":
            # Request modifications, may need to regenerate decision
            return False
        elif action == "escalate":
            # Escalate to higher authority
            return False
        
        return False
    
    def _process_feedback_response(self, interaction: HumanInteraction, 
                                  response: Dict[str, Any]) -> bool:
        """Process feedback response."""
        
        # Feedback is always accepted for learning purposes
        return True
    
    def _process_clarification_response(self, interaction: HumanInteraction, 
                                       response: Dict[str, Any]) -> bool:
        """Process clarification response."""
        
        # Use clarification to improve decision-making
        return True
    
    def _process_escalation_response(self, interaction: HumanInteraction, 
                                    response: Dict[str, Any]) -> bool:
        """Process escalation response."""
        
        # Escalation response provides final authority decision
        return True
    
    def _store_feedback(self, interaction: HumanInteraction, 
                       response: Dict[str, Any]) -> None:
        """Store human feedback for learning."""
        
        feedback = HumanFeedback(
            feedback_id=str(uuid.uuid4()),
            decision_id=interaction.context.get("decision_trace", {}).get("decision_id", ""),
            feedback_type=interaction.context.get("feedback_type", "general"),
            rating=response.get("rating", 3),
            comments=response.get("comments", ""),
            suggested_improvements=response.get("suggested_improvements", []),
            timestamp=datetime.now()
        )
        
        self.feedback_history.append(feedback)
    
    def _format_notification_message(self, interaction: HumanInteraction) -> str:
        """Format notification message for text/email."""
        
        message = f"""
EPPN Alert: {interaction.title}

{interaction.description}

Action Required: {interaction.required_action}

Dashboard: {interaction.dashboard_url}
Chat: {interaction.chat_url}

Priority: {interaction.priority}/5
Created: {interaction.created_at.strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        return message
    
    def _send_notification(self, interaction: HumanInteraction) -> None:
        """Send notification about new interaction."""
        
        # In a real implementation, this would integrate with notification services
        # For now, we'll just log the notification
        
        notification = {
            "interaction_id": interaction.interaction_id,
            "type": interaction.interaction_type.value,
            "title": interaction.title,
            "priority": interaction.priority,
            "dashboard_url": interaction.dashboard_url,
            "chat_url": interaction.chat_url,
            "timestamp": datetime.now().isoformat()
        }
        
        self._log_notification(notification)
    
    def _log_notification(self, notification: Dict[str, Any]) -> None:
        """Log notification for tracking."""
        
        # In a real implementation, this would write to a log file or database
        print(f"Notification: {json.dumps(notification, indent=2)}")
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of human feedback."""
        
        if not self.feedback_history:
            return {"total_feedback": 0}
        
        ratings = [f.rating for f in self.feedback_history]
        
        return {
            "total_feedback": len(self.feedback_history),
            "average_rating": sum(ratings) / len(ratings),
            "rating_distribution": {
                "1": ratings.count(1),
                "2": ratings.count(2),
                "3": ratings.count(3),
                "4": ratings.count(4),
                "5": ratings.count(5)
            },
            "recent_feedback": [
                {
                    "feedback_id": f.feedback_id,
                    "rating": f.rating,
                    "comments": f.comments,
                    "timestamp": f.timestamp.isoformat()
                }
                for f in self.feedback_history[-10:]  # Last 10 feedback items
            ]
        }
    
    def export_interaction_data(self) -> Dict[str, Any]:
        """Export interaction data for analysis."""
        
        return {
            "pending_interactions": [asdict(i) for i in self.pending_interactions.values()],
            "completed_interactions": [asdict(i) for i in self.completed_interactions.values()],
            "feedback_history": [asdict(f) for f in self.feedback_history],
            "summary": {
                "total_pending": len(self.pending_interactions),
                "total_completed": len(self.completed_interactions),
                "total_feedback": len(self.feedback_history)
            }
        }
