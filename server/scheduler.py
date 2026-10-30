"""GPU 작업자 (CALIBRATION.md §0.5).

- GPU 작업자는 1개. 한 번에 최대 batch_max개를 묶어 돌린다
- final이 interim보다 먼저다 (FIFO)
- interim은 세션마다 대기 1개만 둔다. 새 interim이 오면 대기 중인 것을 바꿔 끼운다
  → GPU가 밀리면 interim 간격이 저절로 늘어나고, 대기열이 쌓이지 않는다
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass
class Job:
    key: str  # 세션 키
    kind: str  # interim | final
    audio: object
    future: asyncio.Future
    created: float = field(default_factory=time.perf_counter)


class GpuScheduler:
    def __init__(self, engine, batch_max, interim_max_new_tokens, final_max_new_tokens):
        self.engine = engine
        self.batch_max = batch_max
        self.max_new = {"interim": interim_max_new_tokens, "final": final_max_new_tokens}
        self.finals = []
        self.interims = {}  # key -> Job (도착 순서 유지)
        self.wake = asyncio.Event()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self.task = None
        self.busy_s = 0.0  # 누적 GPU 사용 시간 (healthz, 로그)

    def start(self):
        self.task = asyncio.create_task(self._run())

    async def stop(self):
        if self.task:
            self.task.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def _job(self, key, kind, audio):
        job = Job(key, kind, audio, asyncio.get_running_loop().create_future())
        self.wake.set()
        return job

    def submit_final(self, key, audio):
        job = self._job(key, "final", audio)
        self.finals.append(job)
        return job.future

    def submit_interim(self, key, audio):
        old = self.interims.pop(key, None)
        if old is not None:
            old.future.cancel()
        job = self._job(key, "interim", audio)
        self.interims[key] = job
        return job.future

    def cancel_interim(self, key):
        old = self.interims.pop(key, None)
        if old is not None:
            old.future.cancel()

    def pending(self):
        return len(self.finals) + len(self.interims)

    def _take(self):
        batch = []
        while self.finals and len(batch) < self.batch_max:
            batch.append(self.finals.pop(0))
        for key in list(self.interims):
            if len(batch) >= self.batch_max:
                break
            batch.append(self.interims.pop(key))
        return [j for j in batch if not j.future.cancelled()]

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            if not self.pending():
                self.wake.clear()
                await self.wake.wait()
            batch = self._take()
            if not batch:
                continue
            max_new = max(self.max_new[j.kind] for j in batch)
            started = time.perf_counter()
            try:
                results, timing = await loop.run_in_executor(
                    self.pool, self.engine.transcribe, [j.audio for j in batch], max_new)
            except Exception as e:  # noqa: BLE001 - 작업자는 죽지 않고 해당 작업만 실패시킨다
                for j in batch:
                    if not j.future.done():
                        j.future.set_exception(e)
                continue
            self.busy_s += time.perf_counter() - started
            for j, r in zip(batch, results):
                if not j.future.done():
                    j.future.set_result({**r, **timing, "queue_ms": (started - j.created) * 1000,
                                         "kinds": "+".join(sorted(x.kind[0] for x in batch))})
