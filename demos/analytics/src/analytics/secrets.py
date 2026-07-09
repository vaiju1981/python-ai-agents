"""Secret handling for warehouse connections (PR-3).

- ``SecretProvider`` is a pluggable source of credentials (env by default; a
  vault/secret-manager backend can be dropped in later without touching the
  warehouse adapters).
- ``redact_secrets`` scrubs ``user:password@`` from any string so connection
  URIs never land in logs or provenance envelopes.

Credentials are never embedded in code or config: a warehouse source is built
from a *secret name* that the provider resolves at runtime.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

# Matches a scheme followed by an optional user:password@ section.
_URI_SECRET_RE = re.compile(
    r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)"
    r"(?P<user>[^:/?#@\s]+)"
    r"(?::(?P<password>[^@/?#\s]*))?"
    r"(?P<at>@)"
)


def redact_secrets(text: str) -> str:
    """Replace ``user:password@`` (and ``user@``) in any string with ``***``.

    Idempotent and safe to call on arbitrary text (SQL, logs, provenance).
    """
    if not text:
        return text
    return _URI_SECRET_RE.sub(r"\g<scheme>***:***\g<at>", text)


class SecretProvider(Protocol):
    def get(self, name: str) -> str | None:
        """Return the secret value for ``name``, or ``None`` if unknown."""


class EnvSecretProvider:
    """Resolve secrets from environment variables (default backend)."""

    def get(self, name: str) -> str | None:
        return os.environ.get(name)


def resolve_secret(
    *,
    uri: str | None = None,
    secret_name: str | None = None,
    provider: SecretProvider | None = None,
) -> str:
    """Resolve a warehouse connection URI from either a direct value or a
    named secret via a provider.

    Precedence: explicit ``uri`` → named ``secret_name`` (via provider) → error.
    """
    if uri:
        return uri
    if secret_name:
        backend = provider if provider is not None else EnvSecretProvider()
        value = backend.get(secret_name)
        if not value:
            raise ValueError(f"secret '{secret_name}' not found via {type(backend).__name__}")
        return value
    raise ValueError("either 'uri' or 'secret_name' must be provided")
