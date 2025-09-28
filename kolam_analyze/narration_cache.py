import asyncio
from typing import Any

class AINarrationCache:
    _cache: dict[str, Any] = {}
    _lock = asyncio.Lock()

    @staticmethod
    async def add(key: str, value: Any) -> None:
        async with AINarrationCache._lock:
            AINarrationCache._cache[key] = value

    @staticmethod
    async def get(key: str) -> Any | None:
        return AINarrationCache._cache.get(key)

    @staticmethod
    async def remove(key: str) -> None:
        async with AINarrationCache._lock:
            AINarrationCache._cache.pop(key, None)

    @staticmethod
    async def contains(key: str) -> bool:
        return key in AINarrationCache._cache