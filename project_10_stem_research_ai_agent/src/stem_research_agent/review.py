"""Append-only human-review controls for draft artifacts."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

ALLOWED_ACTIONS = {"approve", "reject", "request_revision", "lock", "supersede"}

@dataclass(frozen=True)
class ReviewEvent:
    event_id: int
    action: str
    reviewer: str
    reason: str
    claim_ids: tuple[str, ...]
    timestamp_utc: str

@dataclass
class ReviewRecord:
    draft_id: str
    events: list[ReviewEvent] = field(default_factory=list)

    def submit(self, action, reviewer, reason, claim_ids=()):
        if action not in ALLOWED_ACTIONS or not reviewer.strip() or not reason.strip():
            raise ValueError("Action, reviewer, and reason are required.")
        if self.is_locked and action != "supersede":
            raise ValueError("Locked drafts require a supersede action before revision.")
        event = ReviewEvent(len(self.events) + 1, action, reviewer, reason, tuple(claim_ids),
                            datetime.now(timezone.utc).isoformat())
        self.events.append(event)
        return event

    @property
    def is_locked(self): return bool(self.events and self.events[-1].action == "lock")

    @property
    def is_approved(self):
        decisions = [
            event.action
            for event in self.events
            if event.action in {"approve", "reject", "request_revision"}
        ]
        return bool(decisions) and decisions[-1] == "approve"

    def exportable(self): return self.is_approved and self.is_locked
    def to_dict(self): return {"draft_id":self.draft_id,"is_locked":self.is_locked,"exportable":self.exportable(),
                               "events":[asdict(event) for event in self.events]}
