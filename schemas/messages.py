"""
Inter-agent message schemas for EPPN

Defines dataclasses for messages exchanged between agents.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class CrawlRequest:
    urls: List[str]
    interpreter_address: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class PDFReady:
    url: str
    source: str
    metadata: Dict[str, Any]


@dataclass
class ParsedText:
    doc_id: str
    sections: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class SummaryReady:
    doc_id: str
    summary: str
    key_points: List[str]
    metadata: Dict[str, Any]


@dataclass
class EthicsReport:
    doc_id: str
    report: Dict[str, Any]
    risks: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


