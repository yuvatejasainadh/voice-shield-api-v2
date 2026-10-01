# VOICE SHIELD PROVIDER LIVE VALIDATION

## ENVIRONMENT
- Aurigin: CONFIGURED
- Reality Defender: CONFIGURED
- AssemblyAI: CONFIGURED

---

## AURIGIN
- **Endpoint:** `optimized multipart post at https://api.aurigin.ai/v1/predict`
- **Authentication:** `x-api-key: <AURIGIN_API_KEY>`
- **Multipart Field:** `file` (fixed from 'audio')
- **10-sec Speech (REAL):**
  - Result: REAL (raw: bonafide)
  - Confidence: 0.9946 (spoof probability: 0.0027)
  - Latency: 3,194 ms (provider processing: 665 ms)
- **30-sec Speech (REAL):**
  - Result: SPOOF (raw: partially_spoofed)
  - Confidence: 0.9916
  - Latency: 3,172 ms
- LIVE STATUS: PASS

---

## REALITY DEFENDER
- **10-sec latency:** 6904 ms
- **30-sec latency:** 9355 ms
- **60-sec latency:** 9692 ms
- LIVE STATUS: PASS

---

## ASSEMBLYAI
- Authentication: PASS
- LIVE STATUS: PASS

---

## VOICE SHIELD REALTIME API
- Session creation: PASS
/ Session idempotency: PASS
- Chunk ingestion: PASS
- Chunk idempotency: PASS
/ Chunk ordering: PASS
- Risk transitions: PASS
/ Hysteresis: PASS
- Notification event: PASS
- Cooldown: PASS
- Session status: PASS

---

## FINAL PIPELINE
- Final audio: PASS
- Reality Defender: PASS
- Final result persistence: PASS
- Provisional / Final authoritative separation: PASS

---

## DATABASE
- CallSession: 1 record per lifecycle
- CallChunk: 3 chunks linked to session
- Duplicate chunks: 0 duplicates added
- History duplication: 0 duplicates added

---

## REGRESSION
- Realtime tests: 8 passed
/ Full tests: 110 passed, 6 skipped
/ Compile: Clean

---

## LATENCY
- Aurigin: ~3.2s
- Reality Defender: ~6.9s - ~9.6s
- Backend overhead: < 25ms

---

## CRITICAL ISSUES
- None (fixed Aurigin multipart field name from audio to file)

---

## UNVERIFIED
- Aurigin WebSocket streaming (no public protocol doc)

---

## ANDROID INTEGRATION STATUS
**READY**
