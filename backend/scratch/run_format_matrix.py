import sys, os, asyncio, time
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from app.services.groq_transcription_service import GroqTranscriptionService
from app.services.reality_defender_service import RealityDefenderService
from app.utils.audio_normalizer import AudioNormalizer

async def test_format(fmt):
    fpath = Path(f'scratch/test_all_formats/sample.{fmt}')
    groq = GroqTranscriptionService()
    rd = RealityDefenderService()
    normalizer = AudioNormalizer()
    
    t0 = time.perf_counter()
    # Test Groq STT
    g_start = time.perf_counter()
    try:
        g_res = await groq.transcribe_file(fpath, include_segments=True)
        g_status = 'PASS'
        g_ms = int((time.perf_counter() - g_start) * 1000)
    except Exception as e:
        g_res = None
        g_status = f'FAIL: {e}'
        g_ms = int((time.perf_counter() - g_start) * 1000)
        
    # Test Reality Defender
    r_start = time.perf_counter()
    normalized_path = None
    try:
        if fmt in ('aac', 'webm', 'm4a'):
            normalized_path = normalizer.normalize_to_wav(fpath)
            r_target = normalized_path
        else:
            r_target = fpath
            
        r_res = await rd.analyze_file(r_target)
        r_status = 'PASS'
        r_ms = int((time.perf_counter() - r_start) * 1000)
    except Exception as e:
        r_res = None
        r_status = f'FAIL: {e}'
        r_ms = int((time.perf_counter() - r_start) * 1000)
    finally:
        if normalized_path and normalized_path.exists():
            normalized_path.unlink(missing_ok=True)
            
    total_ms = int((time.perf_counter() - t0) * 1000)
    return {
        'format': fmt.upper(),
        'groq_status': g_status,
        'rd_status': r_status,
        'groq_ms': g_ms,
        'rd_ms': r_ms,
        'total_ms': total_ms,
        'transcript': getattr(g_res, 'transcript', None),
        'language': getattr(g_res, 'language', None),
        'classification': r_res.get('classification') if isinstance(r_res, dict) else None
    }

async def main():
    formats = ['wav', 'mp3', 'm4a', 'aac', 'ogg', 'flac', 'webm']
    header = f"{'Format':<8} | {'Groq Status':<12} | {'RD Status':<12} | {'Groq ms':<8} | {'RD ms':<8} | {'Total ms':<8} | {'RD Class':<15}"
    print(header)
    print('-' * len(header))
    for fmt in formats:
        res = await test_format(fmt)
        print(f"{res['format']:<8} | {res['groq_status']:<12} | {res['rd_status']:<12} | {res['groq_ms']:<8} | {res['rd_ms']:<8} | {res['total_ms']:<8} | {str(res['classification']):<15}")

asyncio.run(main())
