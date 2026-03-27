class NeuropackError(Exception):
    pass


class MemoryNotFoundError(NeuropackError):
    pass


class DuplicateMemoryError(NeuropackError):
    def __init__(self, existing_id: str):
        self.existing_id = existing_id
        super().__init__(f"Duplicate memory detected, merged into {existing_id}")


class ContentTooLargeError(NeuropackError):
    def __init__(self, message: str = "Content too large", size: int = 0, max_size: int = 0):
        self.size = size
        self.max_size = max_size
        super().__init__(message)


class AuthenticationError(NeuropackError):
    pass


class FormatError(NeuropackError):
    pass


class ValidationError(NeuropackError):
    """Input validation failure with field context."""
    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(f"Validation error on '{field}': {message}")


class FTSQueryError(NeuropackError):
    """FTS5 query syntax error."""
    def __init__(self, query: str, error: str):
        self.query = query
        self.error = error
        super().__init__(f"FTS query error for '{query}': {error}")


class PIIDetectedError(NeuropackError):
    """Content contains sensitive data and PII mode is set to block."""
    def __init__(self, summary: str):
        self.summary = summary
        super().__init__(f"PII blocked: {summary}")


class ContradictionWarning(NeuropackError):
    """New memory contradicts existing memories."""
    def __init__(self, contradictions: list):
        self.contradictions = contradictions
        details = "; ".join(c.reason for c in contradictions[:3])
        super().__init__(f"Contradiction detected: {details}")


class UntrustedSourceError(NeuropackError):
    """Memory from an untrusted source was flagged."""
    def __init__(self, source: str, trust_score: float):
        self.source = source
        self.trust_score = trust_score
        super().__init__(f"Untrusted source '{source}' (score={trust_score:.2f})")


class WorkspaceError(NeuropackError):
    """General workspace operation error."""
    pass


class TaskClaimError(WorkspaceError):
    """Task already claimed by another agent."""
    def __init__(self, task_id: str, assigned_to: str):
        self.task_id = task_id
        self.assigned_to = assigned_to
        super().__init__(f"Task {task_id} already claimed by '{assigned_to}'")
