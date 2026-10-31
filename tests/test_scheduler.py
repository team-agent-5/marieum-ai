"""server/scheduler.py 단위 테스트: 작업마다 출력 토큰 상한. 엔진은 받은 상한을 기록하는 가짜.

  python -m pytest tests -q
"""

import asyncio

from server.scheduler import GpuScheduler


class RecordingEngine:
    def __init__(self):
        self.calls = []

    def transcribe(self, audios, caps):
        self.calls.append(list(caps))
        return [{"text": ""} for _ in audios], {"batch": len(audios)}


def run_one_batch(jobs, rate=14.0, floor=8):
    """jobs: [(kind, vad_speech_ms)]. 작업자가 돌기 전에 모두 넣어 한 묶음으로 돌린다."""
    eng = RecordingEngine()

    async def go():
        sch = GpuScheduler(eng, 4, 96, 128, rate, floor)
        submit = {"final": sch.submit_final, "interim": sch.submit_interim}
        futs = [submit[kind](f"s{i}", None, ms) for i, (kind, ms) in enumerate(jobs)]
        sch.start()
        try:
            return await asyncio.gather(*futs)
        finally:
            await sch.stop()

    results = asyncio.run(go())
    return eng.calls, [r["token_cap"] for r in results]


def test_each_job_gets_its_own_cap_in_one_batch():
    # final 6초: ceil(6 × 14) + 8 = 92, interim 0.5초: ceil(0.5 × 14) + 8 = 15
    calls, caps = run_one_batch([("final", 6000), ("interim", 500)])
    assert calls == [[92, 15]]
    assert caps == [92, 15]


def test_hard_cap_binds_long_speech():
    # ceil(15 × 14) + 8 = 218 → final 고정 상한 128, interim 고정 상한 96
    calls, caps = run_one_batch([("final", 15000), ("interim", 15000)])
    assert calls == [[128, 96]] and caps == [128, 96]


def test_rate_zero_uses_hard_caps():
    calls, caps = run_one_batch([("final", 6000), ("interim", 500)], rate=0)
    assert calls == [[128, 96]] and caps == [128, 96]
