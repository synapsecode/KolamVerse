import asyncio

class KolamFrameManager:
    def __init__(self):
        self._snapshots = []
        self._complete = False
        self._lock = asyncio.Lock()  # ensures atomic updates

    async def clear(self):
        async with self._lock:
            self._snapshots = []
            self._complete = False

    async def add_frame(self, frame):
        async with self._lock:
            self._snapshots.append(frame)

    async def get_frames(self):
        async with self._lock:
            if self._complete and self._snapshots:
                return list(self._snapshots)  # return a copy
            return None

    async def complete_capture(self):
        async with self._lock:
            self._complete = True
