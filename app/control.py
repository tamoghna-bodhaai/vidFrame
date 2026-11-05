from __future__ import annotations


class JobControlError(Exception):
    """Base class for cooperative job control signals."""


class CancelledError(JobControlError):
    """Raised at a checkpoint when the caller cancelled the job."""


class PausedError(JobControlError):
    """Raised at a checkpoint when the caller paused the job."""
