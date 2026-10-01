from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class AudioFormatSchema(BaseModel):
    sampleRate: int = Field(default=16000)
    channels: int = Field(default=1)
    encoding: str = Field(default="pcm_s16le")  # pcm_s16le, wav, aac


class ClientCallStartMessage(BaseModel):
    type: str = Field(default="call_start")
    callSessionId: str = Field(..., min_length=1)
    timestamp: float | None = None


class ClientAudioWindowMessage(BaseModel):
    type: str = Field(default="audio_window")
    callSessionId: str = Field(..., min_length=1)
    sequenceNumber: int = Field(..., ge=1)
    windowStartMs: int = Field(..., ge=0)
    windowEndMs: int = Field(..., ge=0)
    audioFormat: AudioFormatSchema = Field(default_factory=AudioFormatSchema)
    audio: str  # Base64 encoded audio
    clientTimestampMs: float | None = None
    requestId: str | None = None
    windowId: str | None = None
    durationMs: int | None = None


class ClientCallEndMessage(BaseModel):
    type: str = Field(default="call_end")
    callSessionId: str = Field(..., min_length=1)
    timestamp: float | None = None


class LatencyMetricsSchema(BaseModel):
    windowDurationMs: int
    clientTransmissionLatencyMs: int | None = None
    serverQueueLatencyMs: int = 0
    auriginProcessingMs: int = 0
    serverTotalProcessingMs: int = 0
    endToEndLatencyMs: int | None = None
    timeFromCallStartMs: int = 0


class ServerDetectionResultMessage(BaseModel):
    type: str = Field(default="detection_result")
    callSessionId: str
    sequenceNumber: int
    windowStartMs: int
    windowEndMs: int
    result: str  # bonafide, spoofed, partially_spoofed, unknown
    score: float
    confidence: float
    decisionState: str  # observing, suspicious, confirmed, uncertain
    userLabel: str  # ANALYZING, SUSPICIOUS, VOICE CLONING DETECTED, SAFE / LOW RISK, INSUFFICIENT EVIDENCE
    notificationRequired: bool = False
    stateChanged: bool = False
    isEarlyDetection: bool = False
    reason: str | None = None
    processingTimeMs: int = 0
    metrics: LatencyMetricsSchema | None = None
    requestId: str | None = None
    windowId: str | None = None
    durationMs: int | None = None
    serverTimestampMs: float = Field(default_factory=lambda: datetime.now().timestamp() * 1000)


class WindowEvidenceSchema(BaseModel):
    windowId: str
    sequenceNumber: int
    windowStartMs: int
    windowEndMs: int
    rawClassification: str
    normalizedClassification: str
    score: float
    confidence: float
    processingMs: int = 0
    reason: str | None = None
    timestampMs: float = 0.0


class CallReportSchema(BaseModel):
    callSessionId: str
    status: str  # ACTIVE, ANALYZING, RISK_DETECTED, COMPLETED, FAILED, CANCELLED
    startedAt: datetime | str
    endedAt: datetime | str | None = None
    finalClassification: str  # ANALYZING, SAFE / LOW RISK, SUSPICIOUS, VOICE CLONING DETECTED, PARTIALLY_SPOOFED, INSUFFICIENT EVIDENCE
    riskLevel: str  # LOW, MEDIUM, HIGH, UNKNOWN
    confidence: float | None = None
    score: float | None = None
    chunksAnalyzed: int = 0
    totalDurationSeconds: float = 0.0
    evidence: list[WindowEvidenceSchema] = Field(default_factory=list)
    detectorVersion: str = "aurigin-realtime"
    lastUpdatedAt: datetime | str


class ServerCallReportCreatedMessage(BaseModel):
    type: str = Field(default="call_report_created")
    callSessionId: str
    report: CallReportSchema


class ServerCallReportUpdatedMessage(BaseModel):
    type: str = Field(default="call_report_updated")
    callSessionId: str
    report: CallReportSchema
    latestWindow: WindowEvidenceSchema | None = None


class ServerRiskUpdateMessage(BaseModel):
    type: str = Field(default="risk_update")
    callSessionId: str
    report: CallReportSchema


class ServerCallReportCompletedMessage(BaseModel):
    type: str = Field(default="call_report_completed")
    callSessionId: str
    report: CallReportSchema


class ServerSessionClosedMessage(BaseModel):
    type: str = Field(default="session_closed")
    callSessionId: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp() * 1000)


class ServerErrorMessage(BaseModel):
    type: str = Field(default="error")
    callSessionId: str | None = None
    code: str
    message: str


class CreateSessionRequest(BaseModel):
    client_session_id: str = Field(..., min_length=1, max_length=128)
    started_at: datetime | str


class CreateSessionResponse(BaseModel):
    session_id: str
    status: str
    risk_state: str = "LOW"


class ChunkResponse(BaseModel):
    session_id: str
    sequence_number: int
    risk_state: str
    score: float | None = None
    state_changed: bool = False
    notification_required: bool = False
    status: str


class RealtimeRiskResponse(BaseModel):
    risk_state: str
    score: float
    state_changed: bool = False
    notification_required: bool = False


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    current_risk: str = "LOW"
    current_score: float | None = None
    peak_risk: str = "LOW"
    peak_score: float | None = None
    chunks_processed: int = 0
    last_sequence_number: int | None = None
    last_updated_at: datetime | None = None


class CompleteSessionRequest(BaseModel):
    ended_at: datetime | str | None = None
    final_audio_reference: str | None = None


class CompleteSessionResponse(BaseModel):
    session_id: str
    status: str
    final_analysis_id: str | None = None


class FinalAnalysisResponse(BaseModel):
    session_id: str
    status: str
    final_risk: str | None = None
    final_score: float | None = None
    final_provider: str | None = None

