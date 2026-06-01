from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any
from uuid import uuid4


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


@dataclass
class AgentTask:
    query: str
    client_id: str
    task_id: str = field(default_factory=lambda: new_id("task"))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    task_type: str
    selected_agents: list[str]
    selected_tool: str | None
    confidence: float
    reason: str


@dataclass
class ToolResult:
    tool_name: str
    output: str
    success: bool
    latency_ms: float = 0.0


@dataclass
class MemoryRecord:
    memory_id: str
    client_id: str
    memory_type: str
    text: str
    tags: list[str]
    source_task_id: str | None
    sensitivity: str = "private"
    created_at: float = field(default_factory=time)


@dataclass
class ExplanationEvent:
    event_id: str
    task_id: str
    client_id: str
    layer: str
    event_type: str
    decision: str
    reason: str
    confidence: float | None = None
    evidence: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)


@dataclass
class AgentTrace:
    task: AgentTask
    plan: Plan | None = None
    tool_result: ToolResult | None = None
    events: list[ExplanationEvent] = field(default_factory=list)
    user_feedback: str | None = None
    final_answer: str | None = None


@dataclass
class TrainingExample:
    query: str
    label_tool: str
    source_task_id: str


@dataclass
class ClientProfile:
    client_id: str
    capabilities: list[str]
    tools: list[str]
    availability: str = "online"
    load: float = 0.0
    trust_score: float = 1.0


@dataclass
class ClientSelection:
    task_id: str
    selected_client_id: str
    required_capability: str
    score: float
    reason: str
    candidates: list[dict[str, Any]]


@dataclass
class TaskDispatchResult:
    selection: ClientSelection
    trace: AgentTrace
