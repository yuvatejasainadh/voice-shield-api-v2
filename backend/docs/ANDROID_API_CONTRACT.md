# Voice Shield Backend ↔ Android API Contract

**Contract Version**: 2.2.0 (Real-Time In-Call Architecture, Call Report Lifecycle, & Decoupled Detection)  
**Generated Date**: 2026-09-03  
**Backend Architecture**: FastAPI (Asynchronous WebSocket Secure + Aurigin REST + Multi-Window Cumulative Engine + Call Report Lifecycle)  
**Source-of-Truth Files Inspected**:
- `backend/app/main.py`
- `backend/app/api/routes/realtime_ws.py`
- `backend/app/api/routes/realtime.py`
- `backend/app/api/routes/reports.py`
- `backend/app/api/routes/health.py`
- `backend/app/api/routes/ready.py`
- `backend/app/api/routes/analysis.py`
- `backend/app/api/routes/history.py`
- `backend/app/schemas/realtime.py`
- `backend/app/schemas/analysis.py`
- `backend/app/services/detection_decision_engine.py`
- `backend/app/services/realtime_session_manager.py`
- `backend/app/services/aurigin_service.py`
- `backend/app/core/config.py`

---

## Executive Summary & Route Overview

| Category | Count | Status |
|---|---|---|
| **Primary Production Real-Time WebSocket** | 1 (`/api/v1/realtime/ws`, alias `/api/v1/ws`) | **ACTIVE** (Primary In-Call Pipeline) |
| **REST Reports & History** | 2 (`/api/v1/reports`, `/api/v1/reports/{callSessionId}`) | **ACTIVE** (Call Report Retrieval & History) |
| **REST Health & Readiness** | 2 (`/api/v1/health`, `/api/v1/ready`) | **ACTIVE** (Infrastructure Monitoring) |
| **REST Real-Time Fallback & Session** | 3 (`/api/v1/realtime/sessions`, `/sessions/{id}/chunks`, `/sessions/{id}`) | **ACTIVE** (HTTP Fallback / Status) |
| **REST File Analysis & History** | 2 (`/api/v1/analysis`, `/api/v1/history`) | **ACTIVE** (Standalone File Detection & History) |
| **REST Legacy Complete** | 2 (`/api/v1/realtime/sessions/{id}/complete`, `/final-audio`) | **LEGACY / DEPRECATED** |

---

# Part 1: REST API Inventory & Contracts

## REST API Summary Table

| ID | Method | Exact Endpoint Path | Full URL Pattern | Purpose | Android Used? | Backend Status |
|---|---|---|---|---|---|---|
| **REST-001** | `GET` | `/api/v1/health` | `http(s)://<host>:<port>/api/v1/health` | Basic liveness and version check | Optional (Ping) | **ACTIVE** |
| **REST-002** | `GET` | `/api/v1/ready` | `http(s)://<host>:<port>/api/v1/ready` | Readiness and detector dependency check | Optional (Diagnostics) | **ACTIVE** |
| **REST-003** | `GET` | `/api/v1/reports` | `http(s)://<host>:<port>/api/v1/reports` | Paginated call reports with evidence list | **YES (Call History List)** | **ACTIVE** |
| **REST-004** | `GET` | `/api/v1/reports/{call_session_id}` | `http(s)://<host>:<port>/api/v1/reports/{call_session_id}` | Detailed call report and window evidence | **YES (Call Report Detail)** | **ACTIVE** |
| **REST-005** | `POST` | `/api/v1/realtime/sessions` | `http(s)://<host>:<port>/api/v1/realtime/sessions` | Create/register call session in DB | Optional (WSS creates dynamically) | **ACTIVE** |
| **REST-006** | `POST` | `/api/v1/realtime/sessions/{session_id}/chunks` | `http(s)://<host>:<port>/api/v1/realtime/sessions/{session_id}/chunks` | Multipart chunk upload fallback | Fallback only | **ACTIVE** |
| **REST-007** | `GET` | `/api/v1/realtime/sessions/{session_id}` | `http(s)://<host>:<port>/api/v1/realtime/sessions/{session_id}` | Query call session status & peak risk | Optional (Post-call check) | **ACTIVE** |
| **REST-008** | `POST` | `/api/v1/realtime/sessions/{session_id}/complete` | `http(s)://<host>:<port>/api/v1/realtime/sessions/{session_id}/complete` | Close session via REST | NO (WSS `call_end` preferred) | **LEGACY** |
| **REST-009** | `POST` | `/api/v1/realtime/sessions/{session_id}/final-audio` | `http(s)://<host>:<port>/api/v1/realtime/sessions/{session_id}/final-audio` | Legacy post-call full audio upload | **NO (DO NOT CALL)** | **DEPRECATED** |
| **REST-010** | `GET` | `/api/v1/history` | `http(s)://<host>:<port>/api/v1/history` | Standalone file analysis history | Optional | **ACTIVE** |
| **REST-011** | `POST` | `/api/v1/analysis` | `http(s)://<host>:<port>/api/v1/analysis` | Standalone audio file deepfake detector | Optional | **ACTIVE** |


---

## Detailed REST Contracts

### REST-003 — GET `/api/v1/reports`
- **Purpose**: Paginated list of persistent Call Reports for Android Call History screens.
- **Query Parameters**:
  - `page` (`integer`, default: 1, ge: 1)
  - `limit` (`integer`, default: 10, ge: 1, le: 100)
  - `status` (`string`, optional, e.g. `"COMPLETED"`, `"ACTIVE"`)
- **Response Schema** (`200 OK`):
  ```json
  {
    "page": 1,
    "limit": 10,
    "total": 1,
    "items": [
      {
        "callSessionId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
        "status": "COMPLETED",
        "startedAt": "2026-09-03T01:30:00Z",
        "endedAt": "2026-09-03T01:30:30Z",
        "finalClassification": "SAFE / LOW RISK",
        "riskLevel": "LOW",
        "confidence": 0.98,
        "score": 0.05,
        "chunksAnalyzed": 3,
        "totalDurationSeconds": 30.0,
        "evidence": [
          {
            "windowId": "window-001",
            "sequenceNumber": 1,
            "windowStartMs": 0,
            "windowEndMs": 10000,
            "rawClassification": "bonafide",
            "normalizedClassification": "REAL",
            "score": 0.05,
            "confidence": 0.98,
            "processingMs": 45,
            "reason": "Authentic voice sample",
            "timestampMs": 1725300010000
          }
        ],
        "detectorVersion": "aurigin-realtime",
        "lastUpdatedAt": "2026-09-03T01:30:30Z"
      }
    ]
  }
  ```

---

### REST-004 — GET `/api/v1/reports/{call_session_id}`
- **Purpose**: Returns the full Call Report and complete window evidence list for a specific call session.
- **Response Schema** (`200 OK`): Single `CallReportSchema` (same shape as items in `REST-003`).

---

# Part 2: WebSocket Secure (WSS) Contract

This is the **primary production in-call detection and live call report synchronization channel**.

## 1. Connection Details

- **Exact Endpoint**: `/api/v1/realtime/ws` (Alias registered at `/api/v1/ws`)
- **Protocol**: Bidirectional JSON text frames.
- **Heartbeat**: Ping/Pong supported (`{"type": "ping"}` -> `{"type": "pong"}`).

---

## 2. Client → Server Messages

### 1. `call_start`
```json
{
  "type": "call_start",
  "callSessionId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "timestamp": 1725300000000
}
```

### 2. `audio_window`
```json
{
  "type": "audio_window",
  "callSessionId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "sequenceNumber": 1,
  "windowStartMs": 0,
  "windowEndMs": 10000,
  "audioFormat": {
    "sampleRate": 16000,
    "channels": 1,
    "encoding": "pcm_s16le"
  },
  "audio": "<Base64 PCM or WAV>",
  "clientTimestampMs": 1725300010000
}
```

### 3. `call_end`
```json
{
  "type": "call_end",
  "callSessionId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "timestamp": 1725300030000
}
```

---

## 3. Server → Client Messages & Call Report Synchronization

### Event 1: `call_started` & `call_report_created`
Emitted immediately when `call_start` is received:
```json
{
  "type": "call_started",
  "callSessionId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "timestamp": 1725300000045
}
```
Followed by the initial Report:
```json
{
  "type": "call_report_created",
  "callSessionId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "report": {
    "callSessionId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "status": "ACTIVE",
    "startedAt": "2026-09-03T01:30:00Z",
    "endedAt": null,
    "finalClassification": "ANALYZING",
    "riskLevel": "LOW",
    "confidence": null,
    "score": 0.0,
    "chunksAnalyzed": 0,
    "totalDurationSeconds": 0.0,
    "evidence": [],
    "detectorVersion": "aurigin-realtime",
    "lastUpdatedAt": "2026-09-03T01:30:00Z"
  }
}
```

---

### Event 2: `call_report_updated` (and `detection_result`)
Emitted after each audio window is processed by Aurigin and aggregated by the Decision Engine:
```json
{
  "type": "call_report_updated",
  "callSessionId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "report": {
    "callSessionId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "status": "ANALYZING",
    "startedAt": "2026-09-03T01:30:00Z",
    "endedAt": null,
    "finalClassification": "ANALYZING",
    "riskLevel": "LOW",
    "confidence": 0.98,
    "score": 0.05,
    "chunksAnalyzed": 1,
    "totalDurationSeconds": 10.0,
    "evidence": [
      {
        "windowId": "window-001",
        "sequenceNumber": 1,
        "windowStartMs": 0,
        "windowEndMs": 10000,
        "rawClassification": "bonafide",
        "normalizedClassification": "REAL",
        "score": 0.05,
        "confidence": 0.98,
        "processingMs": 45,
        "reason": "Authentic voice sample",
        "timestampMs": 1725300010000
      }
    ],
    "detectorVersion": "aurigin-realtime",
    "lastUpdatedAt": "2026-09-03T01:30:10Z"
  },
  "latestWindow": {
    "windowId": "window-001",
    "sequenceNumber": 1,
    "windowStartMs": 0,
    "windowEndMs": 10000,
    "rawClassification": "bonafide",
    "normalizedClassification": "REAL",
    "score": 0.05,
    "confidence": 0.98,
    "processingMs": 45,
    "reason": "Authentic voice sample",
    "timestampMs": 1725300010000
  }
}
```

---

### Event 3: `risk_update`
Emitted if the accumulated evidence triggers an alert state (`HIGH` or `MEDIUM` risk, or notification required):
```json
{
  "type": "risk_update",
  "callSessionId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "report": {
    "callSessionId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "status": "RISK_DETECTED",
    "startedAt": "2026-09-03T01:30:00Z",
    "endedAt": null,
    "finalClassification": "VOICE CLONING DETECTED",
    "riskLevel": "HIGH",
    "confidence": 0.99,
    "score": 0.96,
    "chunksAnalyzed": 2,
    "totalDurationSeconds": 20.0,
    "evidence": [...],
    "detectorVersion": "aurigin-realtime",
    "lastUpdatedAt": "2026-09-03T01:30:20Z"
  }
}
```

---

### Event 4: `call_report_completed` & `session_closed`
Emitted upon `call_end`:
```json
{
  "type": "call_report_completed",
  "callSessionId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "report": {
    "callSessionId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "status": "COMPLETED",
    "startedAt": "2026-09-03T01:30:00Z",
    "endedAt": "2026-09-03T01:30:30Z",
    "finalClassification": "SAFE / LOW RISK",
    "riskLevel": "LOW",
    "confidence": 0.98,
    "score": 0.05,
    "chunksAnalyzed": 3,
    "totalDurationSeconds": 30.0,
    "evidence": [...],
    "detectorVersion": "aurigin-realtime",
    "lastUpdatedAt": "2026-09-03T01:30:30Z"
  }
}
```
Followed by:
```json
{
  "type": "session_closed",
  "callSessionId": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "timestamp": 1725300030025
}
```

---

## 4. Aurigin Classification Normalization Contract

Aurigin raw prediction results are mapped to standardized Voice Shield classifications as follows:

| Raw Aurigin Classification (`rawClassification`) | Voice Shield Normalized Classification (`normalizedClassification`) | Security Semantics |
|---|---|---|
| `bonafide` / `authentic` / `real` | `REAL` | Voice verified genuine human speech. |
| `spoofed` / `spoof` | `SPOOFED` | Synthetic / AI-generated voice cloning detected. |
| `partially_spoofed` | `SPOOFED` | Presence of spoofed/manipulated audio mixed with genuine human voice. |
| `unknown` / `None` | `UNKNOWN` | Inconclusive or poor audio quality. |

> [!NOTE]
> Aurigin `partially_spoofed` is normalized to Voice Shield `SPOOFED` because it indicates the presence of spoofed/manipulated audio mixed with genuine voice. Raw Aurigin classifications are always preserved in report evidence.


