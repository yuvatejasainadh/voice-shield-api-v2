from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

async def run_test(audio_path: str) -> int:
    api_key = os.getenv('REALITY_DEFENDER_API_KEY')
    if not api_key:
        print('Provider:\nReality Defender\n\nREALITY DEFENDER LIVE TEST FAILED\nReason: REALITY_DEFENDER_API_KEY is not configured in environment.')
        return 1

    path = Path(audio_path)
    if not path.exists():
        print('Audio file not found:', path)
        return 2

    from app.services.reality_defender_service import RealityDefenderService

    service = RealityDefenderService()
    start_time = time.perf_counter()
    try:
        result = await service.analyze_file(path)
        total_latency_ms = int((time.perf_counter() - start_time) * 1000)

        print('Provider:')
        print('Reality Defender\n')
        print('Status:')
        print(result.get('status'), '\n')
        print('Analysis ID:')
        print(result.get('analysis_id'), '\n')
        print('Result:')
        print(result.get('classification'), '\n')
        print('Score:')
        print(result.get('risk_score'), '(ai_probability:', result.get('ai_probability'), ')\n')
        print('Processing latency:')
        print(result.get('processing_time_ms'), 'ms\n')
        print('Total latency:')
        print(total_latency_ms, 'ms')
        return 0
    except Exception as exc:
        total_latency_ms = int((time.perf_counter() - start_time) * 1000)
        print('Provider:\nReality Defender\n')
        print('REALITY DEFENDER LIVE TEST FAILED')
        print('Total latency:', total_latency_ms, 'ms')
        print('Error:', exc)
        return 1

def main() -> int:
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not audio_path:
        print('Usage: python scripts/test_reality_defender.py [audio_path]')
        return 2
    return asyncio.run(run_test(audio_path))

if __name__ == '__main__':
    raise SystemExit(main())
