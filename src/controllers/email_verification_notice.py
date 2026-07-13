from collections.abc import MutableMapping
from typing import Any


def consume_email_verification_notice(
    query_params: MutableMapping[str, Any],
) -> bool:
    marker = query_params.get("verified")
    if marker is None:
        return False

    query_params.pop("verified", None)
    return str(marker) == "true"
