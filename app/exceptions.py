from __future__ import annotations


class QueueFullError(Exception):
    """Raised when a job manager's queue is at capacity and cannot accept new jobs."""
