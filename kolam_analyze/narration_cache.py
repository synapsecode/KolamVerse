import asyncio
from typing import Any

class AINarrationCache:
    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def add(self, key: str, value: Any) -> None:
        async with self._lock:
            self._cache[key] = value

    async def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    async def remove(self, key: str) -> None:
        async with self._lock:
            self._cache.pop(key, None)

    async def contains(self, key: str) -> bool:
        return key in self._cache