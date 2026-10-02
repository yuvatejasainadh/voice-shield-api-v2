# VoiceShield API V2 — Backend

Security-focused backend API for real-time and post-call voice clone & deepfake audio detection, device recording compatibility, Firebase auth, and cybercrime intelligence.

---

## 🔒 Persistence & Production Architecture

VoiceShield V2 uses **PostgreSQL (AWS RDS)** for its persistent data layer, with all schema evolution managed strictly via **Alembic** migrations.

```text
                      Android Client App
                              │
                              │ HTTPS / WSS
                              ▼
                 FastAPI Backend (VoiceShield V2)
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
        Inference Providers             PostgreSQL / AWS RDS
    (Aurigin / Reality Defender)        (psycopg3 / SQLAlchemy V2)
                                             ▲
                                             │
                                      Alembic Migrations
```

### Key Architectural Rules
- **PostgreSQL in Production**: Production deployments require an explicit PostgreSQL connection string (`DATABASE_URL=postgresql+psycopg://...`).
- **Fail-Fast Configuration**: If `ENVIRONMENT=production` is active and `DATABASE_URL` is missing or configured with SQLite, application startup immediately terminates with a configuration error.
- **Alembic Governed**: Production startup does not mutate schemas or call `create_all()`. Migrations are applied via `alembic upgrade head`.
- **SQLite Role**: SQLite is isolated for local offline development and automated testing only.

---

## ⚙️ Environment Configuration

Create a `.env` file in the `backend/` directory based on `.env.example`:

```env
ENVIRONMENT=production
APP_NAME=VoiceShield API V2
APP_VERSION=2.0.0
API_VERSION=v2
API_PREFIX=/api/v2

# PostgreSQL connection string (AWS RDS or local PostgreSQL)
DATABASE_URL=postgresql+psycopg://<APP_USER>:<PASSWORD>@<HOST>:5432/<DATABASE>

# Storage & Upload Limits
MAX_UPLOAD_SIZE_MB=25
STORAGE_PATH=./storage
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Risk Thresholds
RISK_GENUINE_MAX=29
RISK_SUSPICIOUS_MAX=69
MAX_AUDIO_DURATION_SECONDS=600

# Aurigin Real-Time Detection (Primary In-Call Provider)
AURIGIN_API_KEY=your_aurigin_api_key_here
AURIGIN_API_BASE_URL=https://api.aurigin.ai
AURIGIN_TIMEOUT_SECONDS=30
AURIGIN_ENABLED=true
AURIGIN_PROVIDER_VERSION=v1

# Reality Defender RealAPI (Deepfake Analysis)
REALITY_DEFENDER_API_KEY=your_reality_defender_api_key_here
REALITY_DEFENDER_API_BASE_URL=https://api.prd.realitydefender.xyz
REALITY_DEFENDER_TIMEOUT_SECONDS=120
```

---

## 🗄️ Database Management & Migrations

### 1. Apply Migrations to PostgreSQL
```powershell
cd backend
.\.venv\Scripts\alembic.exe upgrade head
```

### 2. Inspect Pending SQL Migrations (Offline Mode)
```powershell
.\.venv\Scripts\alembic.exe upgrade head --sql
```

### 3. Create a New Migration
```powershell
.\.venv\Scripts\alembic.exe revision --autogenerate -m "add_new_columns"
```

### 4. SQLite to PostgreSQL Data Migration Tool
To migrate records from an existing development SQLite database into PostgreSQL:
```powershell
# Dry run simulation
.\.venv\Scripts\python.exe tools/migrate_sqlite_to_postgres.py --source-sqlite voice_clone_detection.db --target-postgres "postgresql+psycopg://user:pass@host:5432/dbname" --dry-run

# Live migration
.\.venv\Scripts\python.exe tools/migrate_sqlite_to_postgres.py --source-sqlite voice_clone_detection.db --target-postgres "postgresql+psycopg://user:pass@host:5432/dbname"
```

Detailed migration runbooks are documented in [`docs/DATABASE_MIGRATION.md`](docs/DATABASE_MIGRATION.md).

---

## 📡 Key API Endpoints

- `POST /api/v2/analyze` — Post-call voice clone and manipulation analysis.
- `POST /api/v2/analysis` — Standalone deepfake analysis submission endpoint.
- `WS /api/v2/realtime/ws` — Real-time streaming WebSocket endpoint for live call chunk evaluation.
- `POST /api/v2/device/compatibility` — Hardware discovery & recording format capability resolver.
- `POST /api/v2/auth/verify-token` — Firebase phone auth token verification & device binding.
- `GET /api/v2/cybercrime/templates` — Fraud/scam intelligence & incident pattern matching.
- `GET /api/v2/health` — Service liveness probe.
- `GET /api/v2/ready` — Component readiness probe (database connectivity and provider configuration).

---

## 🧪 Testing

### Run Automated Mocked Tests
```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest tests -v
```

### Run Real Provider Integration Tests (Optional)
```powershell
$env:RUN_REAL_PROVIDER_TESTS="1"
.\.venv\Scripts\python.exe -m pytest tests/test_real_providers.py -v
```

---

## 🚀 Running the Server Locally

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
