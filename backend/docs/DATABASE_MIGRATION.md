# VoiceShield V2 — Database Architecture & PostgreSQL Migration Guide

## 1. Overview & Target Architecture

VoiceShield V2 utilizes a **PostgreSQL-first** persistence architecture managed strictly via **Alembic** schema migrations.

```text
       ┌────────────────────────────────────────┐
       │     FastAPI Application (VoiceShield)  │
       └───────────────────┬────────────────────┘
                           │
                           ▼
       ┌────────────────────────────────────────┐
       │         SQLAlchemy ORM (V2)            │
       └───────────────────┬────────────────────┘
                           │
                           ▼
       ┌────────────────────────────────────────┐
       │   PostgreSQL 15+ / AWS RDS Instance    │
       └───────────────────▲────────────────────┘
                           │
       ┌───────────────────┴────────────────────┐
       │       Alembic Database Migrations      │
       └────────────────────────────────────────┘
```

### Key Architectural Principles
- **No Runtime `create_all()` in Production**: Application startup never mutates or creates database tables in production. All schema evolution is governed by Alembic.
- **Fail-Fast Production Safeguard**: If `ENVIRONMENT=production` is active and `DATABASE_URL` is missing or configured with an SQLite URI (`sqlite://`), the application immediately fails startup with an explicit configuration error.
- **SQLite Role**: SQLite is strictly development/test-only. It is never used as a silent production fallback.
- **Dialect Consistency**: All JSON fields use PostgreSQL `JSONB` for indexing and query performance, with dialect fallback for test environments.

---

## 2. Environment Configuration

### Production Environment Variables (`.env`)

```ini
ENVIRONMENT=production
APP_NAME=VoiceShield API V2
APP_VERSION=2.0.0

# AWS RDS PostgreSQL connection string (psycopg 3 dialect)
DATABASE_URL=postgresql+psycopg://<APP_USER>:<PASSWORD>@<RDS_ENDPOINT>:5432/<DATABASE_NAME>

# Application settings
MAX_UPLOAD_SIZE_MB=25
STORAGE_PATH=./storage
LOG_LEVEL=INFO
CORS_ORIGINS=https://your-domain.com

# Detection providers
AURIGIN_API_KEY=<AURIGIN_KEY>
AURIGIN_ENABLED=true
REALITY_DEFENDER_API_KEY=<REALITY_DEFENDER_KEY>
```

### Development vs Test Configuration

| Setting | Production | Development | Test |
| :--- | :--- | :--- | :--- |
| `ENVIRONMENT` | `production` | `development` | `test` |
| `DATABASE_URL` | `postgresql+psycopg://...` (Required) | `postgresql+psycopg://...` (Preferred) or local SQLite | `sqlite:///:memory:` (Isolated) |
| Runtime DDL | Prohibited (`Alembic` only) | Alembic (Postgres) or auto-init (SQLite) | Test fixture managed |
| Dialect | PostgreSQL (`JSONB`, `UUID`) | PostgreSQL / SQLite | SQLite memory |

---

## 3. Alembic Migration Workflow

Alembic controls all schema creation, alterations, indexes, and foreign keys.

### 3.1 Migration File Structure
Migrations are located in `backend/alembic/versions/`:
- `0001_initial_schema.py`: Baseline migration containing all 10 V2 tables:
  1. `analysis_records`
  2. `call_sessions`
  3. `call_chunks`
  4. `device_recording_compatibilities`
  5. `users`
  6. `user_devices`
  7. `user_auth_events`
  8. `cybercrime_categories`
  9. `cybercrime_templates`
  10. `cybercrime_template_versions`

### 3.2 Applying Migrations to PostgreSQL
To apply pending migrations to the database configured in `DATABASE_URL`:

```bash
cd backend
alembic upgrade head
```

To view current revision:
```bash
alembic current
```

To inspect generated SQL without connecting (offline mode):
```bash
alembic upgrade head --sql
```

### 3.3 Authoring New Schema Migrations
When models in `app/db/models.py` change:
```bash
alembic revision --autogenerate -m "describe_schema_change"
alembic upgrade head
```
> [!IMPORTANT]
> Never manually edit previously applied historical migrations. Always create new incremental revisions.

---

## 4. SQLite → PostgreSQL Data Migration Procedure

If migrating existing data from a local SQLite database (`voice_clone_detection.db`) to PostgreSQL, use the included idempotent migration utility: `backend/tools/migrate_sqlite_to_postgres.py`.

### Step 4.1: Dry-Run Simulation
Run a non-destructive dry run to validate types, UUID conversions, JSONB structures, and foreign key integrity:

```powershell
cd backend
.\.venv\Scripts\python.exe tools/migrate_sqlite_to_postgres.py `
  --source-sqlite voice_clone_detection.db `
  --target-postgres "postgresql+psycopg://<USER>:<PASS>@<HOST>:5432/<DB>" `
  --dry-run
```

### Step 4.2: Live Migration Execution
Execute the live migration after confirming dry run success:

```powershell
cd backend
.\.venv\Scripts\python.exe tools/migrate_sqlite_to_postgres.py `
  --source-sqlite voice_clone_detection.db `
  --target-postgres "postgresql+psycopg://<USER>:<PASS>@<HOST>:5432/<DB>"
```

### Step 4.3: Verification & Reconciliation Report
The tool verifies row counts for all 10 tables before and after insertion:
```text
Table Name                          | Source   | Read     | Inserted  | Skipped  | Target Total | Status
--------------------------------------------------------------------------------------------------------
analysis_records                    | 0        | 0        | 0         | 0        | 0            | MIGRATED_OK
call_sessions                       | 9        | 9        | 9         | 0        | 9            | MIGRATED_OK
call_chunks                         | 0        | 0        | 0         | 0        | 0            | MIGRATED_OK
device_recording_compatibilities    | 3        | 3        | 3         | 0        | 3            | MIGRATED_OK
users                               | 0        | 0        | 0         | 0        | 0            | MIGRATED_OK
user_devices                        | 0        | 0        | 0         | 0        | 0            | MIGRATED_OK
user_auth_events                    | 0        | 0        | 0         | 0        | 0            | MIGRATED_OK
cybercrime_categories               | 0        | 0        | 0         | 0        | 0            | MIGRATED_OK
cybercrime_templates                | 0        | 0        | 0         | 0        | 0            | MIGRATED_OK
cybercrime_template_versions        | 0        | 0        | 0         | 0        | 0            | MIGRATED_OK
```

---

## 5. Production Deployment Sequence

Follow this exact sequence for production releases:

```text
1. Backup Existing Database (AWS RDS Snapshot / pg_dump)
     │
     ▼
2. Provision / Verify PostgreSQL Instance (AWS RDS)
     │
     ▼
3. Run Alembic Migrations: `alembic upgrade head`
     │
     ▼
4. Verify Schema & Table Definitions
     │
     ▼
5. Optional: Run SQLite -> PostgreSQL Data Migration Utility
     │
     ▼
6. Run Test Suite (`pytest`) Against Test Database
     │
     ▼
7. Set `ENVIRONMENT=production` and `DATABASE_URL` in Secret Manager
     │
     ▼
8. Deploy Backend Application Service (Uvicorn / Gunicorn)
     │
     ▼
9. Verify Health Probe (`GET /api/v2/health`) and Readiness (`GET /api/v2/ready`)
     │
     ▼
10. Retain Rollback Backup Snapshot
```

---

## 6. Backup & Rollback Strategy

### 6.1 AWS RDS Automated Backups & Snapshots
- Enable automated daily snapshots with a minimum 7-day retention period.
- Take a manual snapshot immediately before applying any migration:
  ```bash
  aws rds create-db-snapshot \
    --db-instance-identifier voiceshield-prod-db \
    --db-snapshot-identifier "voiceshield-pre-migration-$(date +%Y%m%d%H%M%S)"
  ```

### 6.2 Rollback Procedure
If a migration or deployment encounters errors:
1. **Schema Rollback via Alembic**:
   ```bash
   alembic downgrade -1
   ```
2. **Database Snapshot Restore**:
   Restore the RDS instance to the pre-migration snapshot if data corruption occurred.
3. **Application Rollback**:
   Re-deploy the previous stable container or package version.
