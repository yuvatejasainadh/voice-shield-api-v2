from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    api_key = os.getenv("AURIGIN_API_KEY")
    if not api_key:
        print("Provider:\nAurigin\n\nAURIGIN LIVE TEST FAILED\nReason: AURIGIN_API_KEY is not configured in environment.")
        return 1

    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not audio_path:
        print("Usage: python scripts/test_aurigin.py <audio_path>")
        return 2

    path = Path(audio_path)
    if not path.exists():
        print(f"Audio file not found: {path}")
        return 3

    base_url = os.getenv("AURIGIN_API_BASE_URL", "https://api.aurigin.ai").rstrip("/")
    url = f"{base_url}/v1/predict"

    start = time.perf_counter()
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                url,
                headers={"x-api-key": api_key},
                files={"file": (path.name, path.read_bytes(), "audio/wav")},
            )
        latency_ms = int((time.perf_counter() - start) * 1000)

        if not response.is_success:
            print("Provider:\nAurigin\n")
            print(f"HTTP status:\n{response.status_code}\n")
            print(f"Request duration:\n{latency_ms} ms\n")
            print("AURIGIN LIVE TEST FAILED")
            print(f"Detail: HTTP {response.status_code} - {response.text[:200]}")
            return 1

        payload = response.json()
        global_block = payload.get("global") or {}
        prediction_id = payload.get("prediction_id") or payload.get("id") or "N/A"
        result = global_block.get("result") or payload.get("result") or "UNKNOWN"
        confidence = global_block.get("confidence") or payload.get("confidence")
        segments = payload.get("segments") or []

        print("Provider:")
        print("Aurigin\n")
        print("HTTP status:")
        print(f"{response.status_code}\n")
        print("Request duration:")
        print(f"{latency_ms} ms\n")
        print("Prediction ID:")
        print(f"{prediction_id}\n")
        print("Global result:")
        print(f"{result}\n")
        print("Confidence:")
        print(f"{confidence}\n")
        print("Segments:")
        print(f"{len(segments)} segments returned")
        return 0

    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        print("Provider:\nAurigin\n")
        print(f"Request duration:\n{latency_ms} ms\n")
        print("AURIGIN LIVE TEST FAILED")
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
