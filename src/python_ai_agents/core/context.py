from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Identity, tenancy, tracing, and deadline context for one agent turn."""

    session_id: str
    principal: str = "anonymous"
    tenant: str = "default"
    trace_id: str | None = None
    deadline: datetime | None = None
    attributes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id is required")
        if self.trace_id is None:
            object.__setattr__(self, "trace_id", self.session_id)
        object.__setattr__(self, "attributes", dict(self.attributes))

    @classmethod
    def ephemeral(cls) -> RequestContext:
        run_id = str(uuid4())
        return cls(session_id=run_id, trace_id=run_id)

    @classmethod
    def session(cls, session_id: str) -> RequestContext:
        return cls(session_id=session_id)

    def child_session(self) -> RequestContext:
        return RequestContext(
            session_id=str(uuid4()),
            principal=self.principal,
            tenant=self.tenant,
            trace_id=self.trace_id,
            deadline=self.deadline,
            attributes=self.attributes,
        )
