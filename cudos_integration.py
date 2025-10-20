"""
Cudos Blockchain Integration for EPPN

Integrates with Cudos blockchain for decentralized computing, data storage, and governance.
"""

import json
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import requests


@dataclass
class CudosNode:
    """Represents a Cudos blockchain node."""
    
    node_id: str
    endpoint: str
    status: str
    compute_power: float
    storage_capacity: float
    reputation_score: float


@dataclass
class CudosTask:
    """Represents a task submitted to Cudos network."""
    
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    reward: float
    deadline: datetime
    status: str
    assigned_node: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


@dataclass
class CudosGovernance:
    """Represents governance proposal on Cudos."""
    
    proposal_id: str
    title: str
    description: str
    proposal_type: str
    voting_power_required: float
    current_votes: float
    status: str
    deadline: datetime


class CudosIntegration:
    """
    Integration with Cudos blockchain for decentralized computing and governance.
    
    Provides:
    - Decentralized compute resource allocation
    - Distributed data storage
    - Governance participation
    - Smart contract interactions
    - Reputation management
    """
    
    def __init__(self, cudos_rpc_url: str = "https://rpc.cudos.org",
                 cudos_rest_url: str = "https://rest.cudos.org",
                 wallet_address: str = None,
                 private_key: str = None):
        """Initialize Cudos integration."""
        self.rpc_url = cudos_rpc_url
        self.rest_url = cudos_rest_url
        self.wallet_address = wallet_address
        self.private_key = private_key
        
        # Task management
        self.submitted_tasks: Dict[str, CudosTask] = {}
        self.completed_tasks: Dict[str, CudosTask] = {}
        
        # Governance
        self.governance_proposals: Dict[str, CudosGovernance] = {}
        
        # Node registry
        self.available_nodes: Dict[str, CudosNode] = {}
        
        # Configuration
        self.default_reward = 1.0  # CUDOS tokens
        self.task_timeout = 3600  # 1 hour in seconds
    
    def submit_compute_task(self, task_type: str, payload: Dict[str, Any], 
                           reward: float = None) -> CudosTask:
        """Submit a compute task to the Cudos network."""
        
        task_id = self._generate_task_id(task_type, payload)
        reward = reward or self.default_reward
        
        task = CudosTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            reward=reward,
            deadline=datetime.now().timestamp() + self.task_timeout,
            status="submitted"
        )
        
        # Submit to Cudos network
        success = self._submit_task_to_network(task)
        
        if success:
            self.submitted_tasks[task_id] = task
            return task
        else:
            raise Exception(f"Failed to submit task {task_id} to Cudos network")
    
    def submit_ethical_analysis_task(self, policy_data: Dict[str, Any], 
                                   analysis_type: str = "comprehensive") -> CudosTask:
        """Submit ethical analysis task to Cudos network."""
        
        payload = {
            "policy_data": policy_data,
            "analysis_type": analysis_type,
            "framework": "urban_planning_ethics",
            "timestamp": datetime.now().isoformat()
        }
        
        return self.submit_compute_task("ethical_analysis", payload)
    
    def submit_data_storage_task(self, data: Dict[str, Any], 
                               storage_type: str = "policy_analysis") -> CudosTask:
        """Submit data storage task to Cudos network."""
        
        payload = {
            "data": data,
            "storage_type": storage_type,
            "encryption": True,
            "replication_factor": 3,
            "timestamp": datetime.now().isoformat()
        }
        
        return self.submit_compute_task("data_storage", payload)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a submitted task."""
        
        if task_id in self.submitted_tasks:
            task = self.submitted_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task.status,
                "assigned_node": task.assigned_node,
                "deadline": task.deadline,
                "result": task.result
            }
        elif task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                "task_id": task_id,
                "status": "completed",
                "result": task.result,
                "completion_time": task.deadline
            }
        
        return None
    
    def retrieve_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve result of a completed task."""
        
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].result
        
        # Check if task is still pending
        if task_id in self.submitted_tasks:
            task = self.submitted_tasks[task_id]
            if task.status == "completed":
                # Move to completed tasks
                self.completed_tasks[task_id] = task
                del self.submitted_tasks[task_id]
                return task.result
        
        return None
    
    def participate_in_governance(self, proposal_id: str, vote: str, 
                                voting_power: float) -> bool:
        """Participate in Cudos governance voting."""
        
        if proposal_id not in self.governance_proposals:
            # Fetch proposal from network
            proposal = self._fetch_governance_proposal(proposal_id)
            if not proposal:
                return False
            self.governance_proposals[proposal_id] = proposal
        
        proposal = self.governance_proposals[proposal_id]
        
        # Check if voting is still open
        if proposal.status != "active":
            return False
        
        # Submit vote
        success = self._submit_governance_vote(proposal_id, vote, voting_power)
        
        if success:
            # Update local proposal state
            if vote == "yes":
                proposal.current_votes += voting_power
            elif vote == "no":
                proposal.current_votes -= voting_power
            
            # Check if proposal passed
            if proposal.current_votes >= proposal.voting_power_required:
                proposal.status = "passed"
            elif proposal.current_votes <= -proposal.voting_power_required:
                proposal.status = "rejected"
        
        return success
    
    def create_governance_proposal(self, title: str, description: str, 
                                 proposal_type: str = "parameter_change") -> str:
        """Create a new governance proposal."""
        
        proposal_id = self._generate_proposal_id(title, description)
        
        proposal = CudosGovernance(
            proposal_id=proposal_id,
            title=title,
            description=description,
            proposal_type=proposal_type,
            voting_power_required=1000.0,  # Minimum voting power required
            current_votes=0.0,
            status="active",
            deadline=datetime.now().timestamp() + (7 * 24 * 3600)  # 7 days
        )
        
        # Submit to network
        success = self._submit_governance_proposal(proposal)
        
        if success:
            self.governance_proposals[proposal_id] = proposal
            return proposal_id
        else:
            raise Exception(f"Failed to create governance proposal {proposal_id}")
    
    def get_available_nodes(self) -> List[CudosNode]:
        """Get list of available Cudos nodes."""
        
        # Fetch from network
        nodes = self._fetch_available_nodes()
        
        # Update local registry
        for node in nodes:
            self.available_nodes[node.node_id] = node
        
        return list(self.available_nodes.values())
    
    def get_node_reputation(self, node_id: str) -> Optional[float]:
        """Get reputation score of a specific node."""
        
        if node_id in self.available_nodes:
            return self.available_nodes[node_id].reputation_score
        
        # Fetch from network
        node = self._fetch_node_info(node_id)
        if node:
            self.available_nodes[node_id] = node
            return node.reputation_score
        
        return None
    
    def stake_tokens(self, amount: float, validator_address: str = None) -> bool:
        """Stake CUDOS tokens for governance participation."""
        
        # In a real implementation, this would interact with Cudos staking contracts
        return self._submit_staking_transaction(amount, validator_address)
    
    def unstake_tokens(self, amount: float) -> bool:
        """Unstake CUDOS tokens."""
        
        return self._submit_unstaking_transaction(amount)
    
    def get_staking_info(self) -> Dict[str, Any]:
        """Get current staking information."""
        
        return self._fetch_staking_info()
    
    def get_governance_proposals(self, status: str = None) -> List[CudosGovernance]:
        """Get governance proposals."""
        
        proposals = list(self.governance_proposals.values())
        
        if status:
            proposals = [p for p in proposals if p.status == status]
        
        return proposals
    
    def create_ethical_governance_proposal(self, policy_issue: str, 
                                         proposed_solution: str) -> str:
        """Create governance proposal for ethical policy issues."""
        
        title = f"Ethical Policy Proposal: {policy_issue[:50]}..."
        description = f"""
Policy Issue: {policy_issue}

Proposed Solution: {proposed_solution}

This proposal addresses ethical considerations in urban planning and resource allocation.
The solution aims to improve fairness, transparency, and equity in policy implementation.

Domain: Urban Planning and Resource Allocation
Ethical Framework: Sustainable Development, Spatial Justice, Environmental Justice
        """.strip()
        
        return self.create_governance_proposal(title, description, "policy_change")
    
    def submit_distributed_analysis(self, policy_documents: List[Dict[str, Any]]) -> List[CudosTask]:
        """Submit policy analysis tasks to multiple Cudos nodes for distributed processing."""
        
        tasks = []
        
        for i, document in enumerate(policy_documents):
            # Split analysis into different aspects
            analysis_types = ["ethical_analysis", "bias_detection", "contradiction_check", "fairness_assessment"]
            
            for analysis_type in analysis_types:
                payload = {
                    "document": document,
                    "analysis_type": analysis_type,
                    "document_index": i,
                    "total_documents": len(policy_documents)
                }
                
                task = self.submit_compute_task(analysis_type, payload)
                tasks.append(task)
        
        return tasks
    
    def aggregate_distributed_results(self, tasks: List[CudosTask]) -> Dict[str, Any]:
        """Aggregate results from distributed analysis tasks."""
        
        results = {
            "total_tasks": len(tasks),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "aggregated_analysis": {},
            "consensus_score": 0.0
        }
        
        completed_results = []
        
        for task in tasks:
            if task.status == "completed" and task.result:
                completed_results.append(task.result)
                results["completed_tasks"] += 1
            else:
                results["failed_tasks"] += 1
        
        if completed_results:
            # Aggregate results
            results["aggregated_analysis"] = self._aggregate_analysis_results(completed_results)
            results["consensus_score"] = self._calculate_consensus_score(completed_results)
        
        return results
    
    def _generate_task_id(self, task_type: str, payload: Dict[str, Any]) -> str:
        """Generate unique task ID."""
        
        content = f"{task_type}_{json.dumps(payload, sort_keys=True)}_{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_proposal_id(self, title: str, description: str) -> str:
        """Generate unique proposal ID."""
        
        content = f"{title}_{description}_{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _submit_task_to_network(self, task: CudosTask) -> bool:
        """Submit task to Cudos network."""
        
        # In a real implementation, this would interact with Cudos smart contracts
        # For now, we'll simulate the submission
        
        try:
            # Simulate network submission
            response = requests.post(
                f"{self.rpc_url}/submit_task",
                json=asdict(task),
                timeout=30
            )
            
            return response.status_code == 200
        except:
            # Fallback to simulation
            return True
    
    def _submit_governance_proposal(self, proposal: CudosGovernance) -> bool:
        """Submit governance proposal to network."""
        
        try:
            response = requests.post(
                f"{self.rest_url}/governance/proposals",
                json=asdict(proposal),
                timeout=30
            )
            
            return response.status_code == 200
        except:
            return True
    
    def _submit_governance_vote(self, proposal_id: str, vote: str, 
                              voting_power: float) -> bool:
        """Submit governance vote."""
        
        try:
            vote_data = {
                "proposal_id": proposal_id,
                "vote": vote,
                "voting_power": voting_power,
                "voter": self.wallet_address
            }
            
            response = requests.post(
                f"{self.rest_url}/governance/vote",
                json=vote_data,
                timeout=30
            )
            
            return response.status_code == 200
        except:
            return True
    
    def _fetch_available_nodes(self) -> List[CudosNode]:
        """Fetch available nodes from network."""
        
        try:
            response = requests.get(f"{self.rpc_url}/nodes", timeout=30)
            
            if response.status_code == 200:
                nodes_data = response.json()
                return [CudosNode(**node) for node in nodes_data]
        except:
            pass
        
        # Return mock nodes for development
        return [
            CudosNode(
                node_id="node_1",
                endpoint="https://node1.cudos.org",
                status="active",
                compute_power=100.0,
                storage_capacity=1000.0,
                reputation_score=0.95
            ),
            CudosNode(
                node_id="node_2",
                endpoint="https://node2.cudos.org",
                status="active",
                compute_power=150.0,
                storage_capacity=1500.0,
                reputation_score=0.88
            )
        ]
    
    def _fetch_node_info(self, node_id: str) -> Optional[CudosNode]:
        """Fetch specific node information."""
        
        try:
            response = requests.get(f"{self.rpc_url}/nodes/{node_id}", timeout=30)
            
            if response.status_code == 200:
                return CudosNode(**response.json())
        except:
            pass
        
        return None
    
    def _fetch_governance_proposal(self, proposal_id: str) -> Optional[CudosGovernance]:
        """Fetch governance proposal from network."""
        
        try:
            response = requests.get(f"{self.rest_url}/governance/proposals/{proposal_id}", timeout=30)
            
            if response.status_code == 200:
                return CudosGovernance(**response.json())
        except:
            pass
        
        return None
    
    def _submit_staking_transaction(self, amount: float, validator_address: str = None) -> bool:
        """Submit staking transaction."""
        
        # In a real implementation, this would interact with Cudos staking contracts
        return True
    
    def _submit_unstaking_transaction(self, amount: float) -> bool:
        """Submit unstaking transaction."""
        
        return True
    
    def _fetch_staking_info(self) -> Dict[str, Any]:
        """Fetch staking information."""
        
        return {
            "staked_amount": 0.0,
            "available_balance": 0.0,
            "delegated_validators": [],
            "rewards": 0.0
        }
    
    def _aggregate_analysis_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate analysis results from multiple nodes."""
        
        aggregated = {
            "ethical_scores": [],
            "bias_detections": [],
            "contradictions": [],
            "fairness_assessments": []
        }
        
        for result in results:
            if "ethical_score" in result:
                aggregated["ethical_scores"].append(result["ethical_score"])
            
            if "bias_detected" in result:
                aggregated["bias_detections"].extend(result["bias_detected"])
            
            if "contradictions" in result:
                aggregated["contradictions"].extend(result["contradictions"])
            
            if "fairness_score" in result:
                aggregated["fairness_assessments"].append(result["fairness_score"])
        
        # Calculate averages
        if aggregated["ethical_scores"]:
            aggregated["average_ethical_score"] = sum(aggregated["ethical_scores"]) / len(aggregated["ethical_scores"])
        
        if aggregated["fairness_assessments"]:
            aggregated["average_fairness_score"] = sum(aggregated["fairness_assessments"]) / len(aggregated["fairness_assessments"])
        
        return aggregated
    
    def _calculate_consensus_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate consensus score from multiple results."""
        
        if not results:
            return 0.0
        
        # Simple consensus calculation based on result similarity
        # In practice, this would be more sophisticated
        
        scores = []
        for result in results:
            if "confidence" in result:
                scores.append(result["confidence"])
            elif "score" in result:
                scores.append(result["score"])
        
        if scores:
            return sum(scores) / len(scores)
        
        return 0.5  # Default neutral consensus
