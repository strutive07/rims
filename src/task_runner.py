import asyncio
import sys
import traceback
from datetime import datetime, timedelta

from tqdm.asyncio import tqdm


class TaskRunner:
    def __init__(self, max_rpm):
        self.semaphore = asyncio.Semaphore(max_rpm)
        self.max_rpm = max_rpm
        self.tasks = []
        self.last_request_times = []

    async def _wrap_task(self, coro):
        async with self.semaphore:
            res = await coro
            # try:
            #     res = await coro
            # except Exception as e:
            #     exc_info = sys.exc_info()
            #     res = {'error': ''.join(traceback.format_exception(*exc_info))}

            return res

    def add_task(self, coro):
        task = self._wrap_task(coro)
        self.tasks.append(task)

    async def run(self):
        res = await tqdm.gather(*self.tasks)

        self.tasks = []
        self.last_request_times = []
        return res
