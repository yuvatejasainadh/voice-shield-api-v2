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
    api_key = os.getenv('ASSEMBLYAI_API_KEY')
    if not api_key:
        print('Provider:\nAssemblyAI\n\nASSEMBLYAI LIVE TEST FAILED\nReason: ASSEMBLYAI_API_KEY is not configured in environment.')
        return 1

    path = Path(audio_path)
    if not path.exists():
        print('Audio file not found:', path)
        return 2

    from app.services.assemblyai_transcription_service import AssemblyAITranscriptionService

    service = AssemblyAITranscriptionService()
    start_time = time.perf_counter()
    try:
        result = await service.transcribe_file(path)
        total_latency_ms = int((time.perf_counter() - start_time) * 1000)

        print('Provider:')
        print('AssemblyAI\n')
        print('Transcript status:')
        print('completed\n')
        print('Detected language:')
        print(result.language, '(probability:', result.language_probability, ')\n')
        print('Speaker count:')
        print(len(result.speakers), '\n')
        print('Transcript length:')
        print(len(result.transcript or ''), 'characters\n')
        print('Diarization segments:')
        print(len(result.speaker_transcript), 'segments\n')
        print('Latency:')
        print(total_latency_ms, 'ms (processing_ms:', result.processing_time_ms, ')')
        return 0
    except Exception as exc:
        total_latency_ms = int((time.perf_counter() - start_time) * 1000)
        print('Provider:\nAssemblyAI\n')
        print('ASSEMBLYAI LIVE TEST FAILED')
        print('Latency:', total_latency_ms, 'ms')
        print('Error:', exc)
        return 1

def main() -> int:
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not audio_path:
        print('Usage: python scripts/test_assemblyai.py [audio_path]')
        return 2
    return asyncio.run(run_test(audio_path))

if __name__ == '__main__':
    raise SystemExit(main())
