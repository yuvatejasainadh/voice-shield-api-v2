"""Direct control script to test and verify the exact Sarvam Batch workflow on audio files."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import httpx

from app.core.config import get_settings

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sarvam-exact-test")


async def main() -> None:
    settings = get_settings()
    api_key = settings.sarvam_api_key
    if not api_key:
        print("ERROR: SARVAM_API_KEY is not set.")
        return

    target_file = Path("storage/audio_8e5e088a977c449783f6f1d540d1e460.mp3")
    if not target_file.exists():
        print(f"ERROR: {target_file} does not exist.")
        return

    audio_bytes = target_file.read_bytes()
    sha256_hash = hashlib.sha256(audio_bytes).hexdigest()
    content_type = "audio/mpeg"

    print("==================================================================")
    print("SARVAM DIRECT CONTROL WORKFLOW TEST")
    print("==================================================================")
    print(f"File: {target_file.name}")
    print(f"Size: {len(audio_bytes)} bytes")
    print(f"SHA256: {sha256_hash}")
    print(f"Content-Type: {content_type}")

    headers = {"api-subscription-key": api_key}
    base_url = "https://api.sarvam.ai"

    # Step 1: Init Job
    init_payload = {
        "job_parameters": {
            "model": "saaras:v4",
            "language_code": "unknown",
            "mode": "codemix",
            "with_timestamps": True,
            "with_diarization": True,
        }
    }
    print("\n1. Initializing Job with payload:")
    print(json.dumps(init_payload, indent=2))

    async with httpx.AsyncClient(timeout=60.0) as client:
        t0 = time.perf_counter()
        init_resp = await client.post(f"{base_url}/speech-to-text/job/v1", headers=headers, json=init_payload)
        print(f"Init status: {init_resp.status_code}")
        init_data = init_resp.json()
        job_id = init_data["job_id"]
        print(f"Job ID: {job_id}")

        # Step 2: Upload Files URL
        upload_resp = await client.post(
            f"{base_url}/speech-to-text/job/v1/upload-files",
            headers=headers,
            json={"job_id": job_id, "files": [target_file.name]},
        )
        print(f"Upload URL request status: {upload_resp.status_code}")
        upload_data = upload_resp.json()
        azure_upload_url = upload_data["upload_urls"][target_file.name]["file_url"]

        # Step 3: Azure Blob PUT
        azure_headers = {
            "x-ms-blob-type": "BlockBlob",
            "Content-Type": content_type,
        }
        put_resp = await client.put(azure_upload_url, content=audio_bytes, headers=azure_headers, timeout=60.0)
        print(f"Azure PUT status: {put_resp.status_code}")

        # Step 4: Start Job
        start_resp = await client.post(f"{base_url}/speech-to-text/job/v1/{job_id}/start", headers=headers)
        print(f"Job Start status: {start_resp.status_code}")

        # Step 5: Poll Status
        poll_states = []
        for i in range(30):
            await asyncio.sleep(2)
            st_resp = await client.get(f"{base_url}/speech-to-text/job/v1/{job_id}/status", headers=headers)
            st_data = st_resp.json()
            state = st_data.get("job_state")
            poll_states.append(state)
            print(f"Poll #{i+1}: {state}")
            if state in ("Completed", "Failed", "Cancelled"):
                break

        # Step 6: Download Files
        dl_resp = await client.post(
            f"{base_url}/speech-to-text/job/v1/download-files",
            headers=headers,
            json={"job_id": job_id, "files": ["0.json"]},
        )
        print(f"Download request status: {dl_resp.status_code}")
        dl_data = dl_resp.json()
        download_url = dl_data["download_urls"]["0.json"]["file_url"]

        # Step 7: Get Raw JSON
        raw_resp = await client.get(download_url)
        raw_json = raw_resp.json()
        total_time = int((time.perf_counter() - t0) * 1000)

        print("\n==================================================================")
        print("SARVAM DIRECT CONTROL SUMMARY RESULTS")
        print("==================================================================")
        transcript = raw_json.get("transcript") or ""
        timestamps = raw_json.get("timestamps") or {}
        entries = raw_json.get("diarized_transcript", {}).get("entries", [])
        lang_detected = raw_json.get("language_code")
        lang_prob = raw_json.get("language_probability")

        print(f"Total Time: {total_time}ms")
        print(f"Transcript Length: {len(transcript)} chars")
        print(f"Detected Language: {lang_detected} (prob={lang_prob})")
        print(f"Timestamps count: {len(timestamps.get('words', [])) if isinstance(timestamps, dict) else 0}")
        print(f"Diarized entries count: {len(entries)}")
        print(f"Transcript preview: {transcript[:300]}...")


if __name__ == "__main__":
    asyncio.run(main())
