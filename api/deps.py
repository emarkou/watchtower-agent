from __future__ import annotations

from typing import Optional

from agent.run_store import RunStore

_store: Optional[RunStore] = None


def set_store(store: RunStore) -> None:
    global _store
    _store = store


def get_store() -> RunStore:
    if _store is None:
        raise RuntimeError("RunStore not initialised — lifespan may not have run")
    return _store
