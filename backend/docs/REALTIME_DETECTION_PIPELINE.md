# Voice Shield — Production Real-Time Detection Pipeline

## 1. Architecture Overview

```
Android App (Call Audio Capture - PCM 16kHz 16-bit Mono)
       |
       |  WebSocket Secure (WSS: /api/v1/realtime/ws)
       |  JSON frames: call_start, audio_window, call_end
       v
Voice Shield Server (FastAPI WSS Gateway)
       |
       |  1. Validate active callSessionId & window order
       |  2. Reconstruct original audio PCM/WAV buffer
       |  3. Submit asynchronously to Aurigin REST API
       v
Aurigin Detection Service (REST: /v1/predict)
       |  Credentials kept strictly server-side
       |  Returns calibrated score (0.0..1.0), confidence, segments
       v
Voice Shield Server
       |  4. Normalize Aurigin output (bonafide, spoofed, partially_spoofed)
       |  5. Process through DetectionDecisionEngine (multi-window evidence, early spoof detection)
       |  6. Check session state (ACTIVE vs CLOSED)
       |  7. Track granular latency metrics
       |  8. Emit detection_result over WSS
       v
Android App
       - If callSessionId == activeCallSessionId: display alert & waveform status
       - If callSessionId != activeCallSessionId: discard stale result silently
```

---

## 2. Privacy & Audio Signal Integrity Principles

1. **No Transcription**: Transcription (STT, diarization, language detection) is completely removed from the clone detection pipeline.
2. **No VAD / Silence Removal**: Original call audio waveform, PCM characteristics, timing, and speech/silence segments are preserved.
3. **No Server-Side Persistence**: Voice Shield server does not save or persist raw user call audio or transcripts.
4. **No Post-Call Full-Audio Analysis**: Call sessions transition to `CLOSED` upon hangup, and all post-call full-audio detection dependencies and notifications are eliminated.

---

## 3. WebSocket Protocol (WSS)

### Endpoint
`wss://<server-domain>/api/v1/realtime/ws` (or `ws://localhost:8000/api/v1/realtime/ws` in dev)

### Message Flow

#### 1. Call Start (Client -> Server)
```json
{
  "type": "call_start",
  "callSessionId": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1725300000000
}
```

#### Server Ack (Server -> Client)
```json
{
  "type": "call_started",
  "callSessionId": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1725300000050
}
```

#### 2. Audio Window (Client -> Server)
Sent on cumulative schedule (0–10s, 0–20s, 0–30s, etc.):
```json
{
  "type": "audio_window",
  "callSessionId": "550e8400-e29b-41d4-a716-446655440000",
  "sequenceNumber": 1,
  "windowStartMs": 0,
  "windowEndMs": 10000,
  "audioFormat": {
    "sampleRate": 16000,
    "channels": 1,
    "encoding": "pcm_s16le"
  },
  "audio": "<Base64 encoded raw PCM or WAV>",
  "clientTimestampMs": 1725300010000
}
```

#### 3. Detection Result (Server -> Client)
```json
{
  "type": "detection_result",
  "callSessionId": "550e8400-e29b-41d4-a716-446655440000",
  "sequenceNumber": 1,
  "windowStartMs": 0,
  "windowEndMs": 10000,
  "result": "bonafide",
  "score": 0.05,
  "confidence": 0.98,
  "decisionState": "observing",
  "userLabel": "ANALYZING",
  "notificationRequired": false,
  "stateChanged": false,
  "isEarlyDetection": false,
  "reason": "Analyzing call audio (0–10s)...",
  "processingTimeMs": 65,
  "metrics": {
    "windowDurationMs": 10000,
    "clientTransmissionLatencyMs": 42,
    "serverQueueLatencyMs": 2,
    "auriginProcessingMs": 58,
    "serverTotalProcessingMs": 65,
    "endToEndLatencyMs": 107,
    "timeFromCallStartMs": 10107
  },
  "serverTimestampMs": 1725300010107
}
```

#### 4. Call End (Client -> Server)
```json
{
  "type": "call_end",
  "callSessionId": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1725300035000
}
```

#### Server Session Closed (Server -> Client)
```json
{
  "type": "session_closed",
  "callSessionId": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1725300035020
}
```

---

## 4. DetectionDecisionEngine Logic

- **Cumulative Windowing**: Aggregates 0–10s, 0–20s, 0–30s+ window observations with linear duration weighting.
- **Early Detection**: If window 1 (0–10s) exhibits spoof score $\ge 0.85$ with confidence $\ge 0.60$, transitions directly to `CONFIRMED` ("VOICE CLONING DETECTED") early.
- **Mixed-Audio Resilience**: A single 10–20s anomaly flags state as `SUSPICIOUS` but requires cumulative confirmation before escalating to confirmed alert.
- **Decision States & Labels**:
  - `observing` -> `ANALYZING`
  - `suspicious` -> `SUSPICIOUS`
  - `confirmed` (high spoof) -> `VOICE CLONING DETECTED`
  - `confirmed` (low spoof after 30s) -> `SAFE / LOW RISK`
  - `uncertain` -> `INSUFFICIENT EVIDENCE`

---

## 5. Session Lifecycle & Notification Safety

1. Every active call generates a UUID `callSessionId`.
2. All WSS communication tags `callSessionId`.
3. When Call A ends, server marks session as `CLOSED` and cancels pending asynchronous tasks.
4. Any delayed result arriving after session closure is dropped immediately.
5. Android client validates incoming `callSessionId == activeCallSessionId` before updating UI or firing in-call notifications, ensuring Call A results can never trigger a notification during Call B.
