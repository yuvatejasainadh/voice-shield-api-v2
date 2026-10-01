# Voice Clone Detection Backend

Security-focused backend API for voice anti-spoofing, deepfake detection, and conversational speech-to-text transcription.

---

## 🔒 Production Architecture

The backend leverages cloud-accelerated inference providers for high-speed, reliable analysis:

```text
                     Android App
                          │
                          │ HTTPS
                          ▼
              voice-api.asteriontechnologies.tech
                          │
                          ▼
                   FastAPI Backend
                          │
                   Audio Validation
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
        Groq API                  Reality Defender
    (whisper-large-v3)                RealAPI
    Speech-to-Text             Deepfake / AI Detection
             │                         │
             ▼                         ▼
         Transcript              Manipulation
         Timestamps               Probability
          Language              Classification
             │                         │
             └────────────┬────────────┘
                          ▼
                 Normalized Response
                          │
                          ▼
                     Android App
```

---

## 🌐 External Processing & Privacy Notice

> [!IMPORTANT]
> **Privacy & Data Safety Notice**:
> Call and voice audio uploaded to this backend is transmitted securely over HTTPS to external third-party processing providers:
> 1. **Groq Inc.** (`api.groq.com`) — Speech-to-text transcription.
> 2. **Reality Defender** (`api.prd.realitydefender.xyz`) — Synthetic speech and deepfake manipulation detection.
>
> **Security Guardrails**:
> - Provider API keys are strictly server-side environment variables and are **never exposed to client applications**.
> - Mobile clients (Android) communicate exclusively with our backend (`voice-api.asteriontechnologies.tech`).
> - Play Store Data Safety and Privacy Policies must accurately disclose that audio data is processed via external AI partners.

---

## 🚀 Providers & Capabilities

### 1. Groq Speech-to-Text
- **Model**: `whisper-large-v3` (OpenAI-compatible STT endpoint)
- **Endpoint**: `POST https://api.groq.com/openai/v1/audio/transcriptions`
- **Output**: Full transcript, detected language, and timestamped segments (`start`, `end`, `text`).
- **Multilingual**: High accuracy on **Telugu** (`te`), **English** (`en`), Indian accents, and code-switched **Tinglish**.

### 2. Reality Defender RealAPI
- **SDK**: Official `realitydefender` Python SDK (`RealityDefender(api_key=...)`)
- **Workflow**: Async signed upload followed by bounded result polling.
- **Output**: Manipulation probability score (0.0–1.0), risk classification, sub-model explainability indicators.
- **Classification Mapping**:
  - `0–29%` manipulation probability: `LIKELY_GENUINE`
  - `30–69%` manipulation probability: `SUSPICIOUS`
  - `70–100%` manipulation probability: `LIKELY_AI_GENERATED`

### 3. Speaker Diarization
- Neither Groq nor Reality Defender currently provides speaker diarization.
- Diarization fields return `"speakers": []` and `"speaker_transcript": []`. Speaker labels are never fabricated.

---

## ⚙️ Environment Configuration

Create a `.env` file in the `backend/` directory:

```env
APP_NAME=Voice Clone Detection API
APP_VERSION=0.1.0
DATABASE_URL=sqlite:///./voice_clone_detection.db
MAX_UPLOAD_SIZE_MB=25
STORAGE_PATH=./storage
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Risk Thresholds
RISK_GENUINE_MAX=29
RISK_SUSPICIOUS_MAX=69

# Groq Speech-to-Text
GROQ_API_KEY=your_groq_api_key_here
GROQ_TRANSCRIPTION_MODEL=whisper-large-v3
GROQ_API_BASE_URL=https://api.groq.com/openai/v1
GROQ_API_TIMEOUT_SECONDS=60

# Reality Defender RealAPI
REALITY_DEFENDER_API_KEY=your_reality_defender_api_key_here
REALITY_DEFENDER_API_BASE_URL=https://api.prd.realitydefender.xyz
REALITY_DEFENDER_TIMEOUT_SECONDS=120
```

---

## 📡 API Endpoints

### 1. `POST /api/v1/analyze` (Main Mobile Endpoint)
Upload audio for concurrent transcription (Groq) and voice clone detection (Reality Defender):

**Request**:
`multipart/form-data` with `audio=<binary_data>` and optional `language=te`.

**Response (200 OK)**:
```json
{
  "success": true,
  "transcription": {
    "text": "మీరు ఎక్కడ ఉన్నారు?",
    "language": "telugu",
    "language_probability": null,
    "duration_seconds": 3.2
  },
  "speakers": [],
  "speaker_transcript": [],
  "voice_analysis": {
    "analysis_id": "c1f7a2d8-...",
    "status": "completed",
    "classification": "LIKELY_GENUINE",
    "risk_score": 12,
    "confidence": null,
    "ai_probability": 0.12,
    "duration_seconds": 3.2,
    "segments_analyzed": 1,
    "processing_time_ms": 1420,
    "detector_version": "reality-defender",
    "reasons": ["No significant deepfake or synthetic voice manipulation detected"],
    "created_at": "2026-08-29T22:00:00"
  },
  "processing_time_ms": 1450,
  "processing": {
    "transcription_ms": 820,
    "voice_analysis_ms": 1420,
    "total_ms": 1450
  }
}
```

### 2. `POST /api/v1/transcription`
Speech-to-text transcription via Groq:
- `audio`: audio file (.wav, .mp3, .m4a, .aac, .ogg, .flac, .webm)
- `language`: optional language code (e.g. `te`, `en`)
- `include_segments`: boolean (include timestamped segments)

### 3. `POST /api/v1/analyze`
Combined cloud analysis via Sarvam, AssemblyAI, Groq, and Reality Defender. Returns transcription, speaker diarization, provider status, and normalized voice analysis.

### 4. `GET /api/v1/ready`
Reports status of database and provider configurations:
```json
{
  "status": "ready",
  "database": true,
  "groq": "configured",
  "reality_defender": "configured",
  "groq_model": "whisper-large-v3",
  "detector_provider": "reality-defender"
}
```

---

## 🧪 Testing

### 1. Run Automated Mocked Tests (Zero external network calls)
```powershell
.\.venv\Scripts\python.exe -m pytest
```

### 2. Run Real Provider Integration Tests (Optional)
Requires `GROQ_API_KEY` and `REALITY_DEFENDER_API_KEY` set in `.env`:
```powershell
$env:RUN_REAL_PROVIDER_TESTS="1"
.\.venv\Scripts\python.exe -m pytest tests/test_real_providers.py -v
```

---

## 🚀 Running the Server Locally

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
