import re
from dataclasses import dataclass
from typing import Any

from supabase import Client


class ValidationError(ValueError):
    """Safe validation message that can be shown to the user."""


class AuthError(RuntimeError):
    """Safe authentication message that does not expose provider internals."""


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str
    email: str
    full