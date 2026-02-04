from __future__ import annotations

_verbose = False


def set_verbose(enabled: bool) -> None:
    global _verbose
    _verbose = enabled


def is_verbose() -> bool:
    return _verbose


def log(message: str) -> None:
    if _verbose:
        print(message)
