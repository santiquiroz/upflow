from __future__ import annotations


class QueueFullError(Exception):
    """Raised when a job manager's queue is at capacity and cannot accept new jobs."""


class HfDownloadError(Exception):
    """Base error for Hugging Face model download failures."""


class HfDownloadTooLargeError(HfDownloadError):
    """Raised when a download exceeds settings.max_model_download_mb."""


class HfInvalidSourceError(HfDownloadError):
    """Raised when a download URL is not HTTPS huggingface.co."""


class ModelNotFoundError(Exception):
    """Raised when a model id does not exist in the registry."""


class ModelProtectedError(Exception):
    """Raised when attempting to delete a builtin (non-removable) model."""
