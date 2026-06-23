__version__ = "1.2.0"

from .client import (
    Client,
    PullProgress,
    FreeAIAgentError,
    ServerNotRunning,
    BackendUnavailable,
    DownloadInProgress,
)

__all__ = [
    "__version__",
    "Client",
    "PullProgress",
    "FreeAIAgentError",
    "ServerNotRunning",
    "BackendUnavailable",
    "DownloadInProgress",
]
