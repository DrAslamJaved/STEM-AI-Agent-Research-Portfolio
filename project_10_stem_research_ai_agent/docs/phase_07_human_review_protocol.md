# Phase 7 Human Review Protocol

Review actions are append-only: approve, reject, request revision, lock, and
supersede. A locked draft rejects further actions until an explicit supersede
event records why it was reopened. Export requires both approval and a final
lock. Review records retain reviewer, rationale, claim IDs, and UTC timestamp.
