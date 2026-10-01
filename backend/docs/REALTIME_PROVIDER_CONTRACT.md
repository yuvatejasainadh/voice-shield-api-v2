# VOICE SHIELD - REALTIME PROVIDER & BACKEND API CONTRACT

## 1. Architecture Overview
Voice Shield provides continuous, low-latency deepfake detection during active calls combined with authoritative post-call verification.

- **Realtime Path:** 10s audio chunks -> FastAPI -> Aurigin REST -> Rolling Hysteresis Engine -> Android Event
- **Authoritative Post-Call Path:** Full recording -> Reality Defender RealAPI -> CallSession Completion

## 2. Android -> FastAPI Contract
- POST /api/v1/realtime/sessions: Idempotent session creation
- POST /api/v1/realtime/sessions/{session_id}/chunks: Multipart chunk upload with sequence number and idempotency key
- GET /api/v1/realtime/sessions/{session_id}: Realtime session status
- POST /api/v1/realtime/sessions/{session_id}/complete: Session completion trigger
- POST /api/v1/realtime/sessions/{session_id}/final-audio: Authoritative audio upload

## 3. Provider Protocols & Authentication
- **Aurigin:** POST /v1/predict with x-api-key. Returns global.result (REAL/SPOOF), confidence. WebSocket: UNVERIFIED.
- **Reality Defender:** RealAPI SDK upload + polling with x-api-key. Output: AUTHENTIC/MANIPULATED, score.
- **AssemblyAI:** v2 REST with Authorization: <key>. Parameters: speaker_labels=True, speakers_expected=2, language_detection=True.

## 4. Risk Engine & Notification Rules
- States: LOW, MEDIUM, HIGH, UNKNOWN.
- Window size: 5 observations. Medium threshold: 0.55. High threshold: 0.80.
- Alerts trigger strictly on LOW -> MEDIUM and MEDIUM -> HIGH transitions with 30s cooldown.
