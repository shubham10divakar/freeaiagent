"""Caller/session resolution.

Lets apps maintain separate context threads with zero config. Resolution order:

  1. explicit `session_id` in the request body (wins)
  2. `X-Caller-ID` request header (set once per app, then forget)
  3. `"default"` session (fallback)

Port-fingerprint detection from the original design is intentionally omitted —
ephemeral client ports change per request, so it is unreliable. The header is
the solid, recommended mechanism.
"""
from typing import Optional

CALLER_HEADER = "X-Caller-ID"
DEFAULT_SESSION = "default"


def resolve_session(body_session_id: Optional[str], header_value: Optional[str]) -> str:
    """Resolve the effective session id from a request body value + header value."""
    if body_session_id and body_session_id != DEFAULT_SESSION:
        return body_session_id
    if header_value and header_value.strip():
        return header_value.strip()
    return DEFAULT_SESSION
