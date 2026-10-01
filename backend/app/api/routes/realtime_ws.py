"""WebSocket Secure endpoint for real-time in-call voice clone detection."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
import uuid
import wave
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.pipeline_logger import VoiceShieldPipelineLogger, format_call_id, format_window_id
from app.schemas.realtime import (
    ClientAudioWindowMessage,
    ClientCallEndMessage,
    ClientCallStartMessage,
    LatencyMetricsSchema,
    ServerDetectionResultMessage,
    ServerErrorMessage,
    ServerSessionClosedMessage,
)
from app.services.aurigin_service import AuriginService
from app.services.realtime_session_manager import RealtimeSessionManager

router = APIRouter(tags=["realtime-ws"])
logger = logging.getLogger("voice-clone-detection")


def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw 16-bit PCM samples into standard WAV container bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buf.getvalue()


@router.websocket("/realtime/ws")
@router.websocket("/ws")
async def realtime_detection_websocket(websocket: WebSocket) -> None:
    """
    Bidirectional WebSocket endpoint for in-call real-time voice-clone detection.
    
    Protocol:
    - Client -> Server: call_start, audio_window, call_end, ping
    - Server -> Client: detection_result, session_closed, pong, error
    """
    await websocket.accept()
    settings = get_settings()
    session_manager = RealtimeSessionManager.get_instance()
    aurigin_service = AuriginService()
    active_call_session_id: str | None = None

    logger.info("WebSocket client connected")

    # Helper function to process and emit a window
    async def _process_and_emit_window(
        sess_id: str,
        seq_num: int,
        win_id: str,
        win_start_ms: int,
        win_end_ms: int,
        resolved_dur_ms: int,
        audio_payload: bytes,
        server_recv_ms: float,
        resolved_req_id: str,
        ws: WebSocket,
        client_timestamp_ms: float | None = None,
    ) -> None:
        server_proc_start = time.perf_counter()

        # Send to Aurigin REST with full correlation
        aurigin_t0 = time.perf_counter()
        VoiceShieldPipelineLogger.detector_started(window_id=win_id, detector="AURIGIN", request_id=resolved_req_id)
        try:
            provider_res = await aurigin_service.analyze_audio(
                audio_payload,
                filename=f"window_{seq_num}.wav",
                content_type="audio/wav",
                call_id=sess_id,
                request_id=resolved_req_id,
                window_id=win_id,
                duration_ms=resolved_dur_ms,
            )
        except Exception as exc:
            logger.exception("Aurigin REST call exception: %s", exc)
            provider_res = {
                "provider": "aurigin",
                "prediction_id": None,
                "result": "UNKNOWN",
                "score": 0.45,
                "confidence": 0.0,
                "processing_ms": int((time.perf_counter() - aurigin_t0) * 1000),
                "raw_provider_status": "ERROR",
                "raw_result": "unknown",
                "reason": f"Provider error: {exc}",
            }

        aurigin_ms = int((time.perf_counter() - aurigin_t0) * 1000)
        raw_res_val = provider_res.get("raw_result")
        norm_res_val = provider_res.get("result", "UNKNOWN")
        score_val = provider_res.get("score")
        conf_val = provider_res.get("confidence")
        reason_val = provider_res.get("reason")

        # Emit detector completed log
        if raw_res_val == "partially_spoofed":
            res_label = "PARTIALLY_SPOOFED"
        elif norm_res_val == "REAL" or raw_res_val == "bonafide":
            res_label = "BONAFIDE"
        elif norm_res_val in ("SPOOF", "SPOOFED") or raw_res_val == "spoofed":
            res_label = "SPOOF"
        else:
            res_label = "UNKNOWN"

        VoiceShieldPipelineLogger.detector_completed(
            window_id=win_id,
            result=res_label,
            confidence=conf_val,
            score=score_val,
            latency_ms=aurigin_ms,
            prediction_id=provider_res.get("prediction_id"),
        )

        # Verify session exists before applying decision & sending result
        sess = session_manager.get_session(sess_id)
        if sess is None:
            logger.info(
                "Session %s was not found while Aurigin finished sequence %d; discarding",
                sess_id,
                seq_num,
            )
            return

        # Feed into Decision Engine under session lock
        async with sess.lock:
            decision_output = sess.decision_engine.add_observation(
                sequence_number=seq_num,
                window_start_ms=win_start_ms,
                window_end_ms=win_end_ms,
                raw_result=raw_res_val,
                normalized_result=norm_res_val,
                score=score_val,
                confidence=conf_val,
                processing_ms=aurigin_ms,
                reason=reason_val,
                timestamp_ms=server_recv_ms,
                window_id=win_id,
            )

            now_ms = time.time() * 1000
            server_total_ms = int((time.perf_counter() - server_proc_start) * 1000)
            window_dur_ms = win_end_ms - win_start_ms
            time_from_start = int(now_ms - sess.created_at_ms)

            client_trans_lat = None
            e2e_latency = None
            if client_timestamp_ms is not None:
                client_trans_lat = max(0, int(server_recv_ms - client_timestamp_ms))
                e2e_latency = max(0, int(now_ms - client_timestamp_ms))

            metrics = LatencyMetricsSchema(
                windowDurationMs=window_dur_ms,
                clientTransmissionLatencyMs=client_trans_lat,
                serverQueueLatencyMs=max(0, int(server_proc_start * 1000 - server_recv_ms)),
                auriginProcessingMs=aurigin_ms,
                serverTotalProcessingMs=server_total_ms,
                endToEndLatencyMs=e2e_latency,
                timeFromCallStartMs=time_from_start,
            )

            # Track first meaningful detection
            if sess.first_detection_ms is None and decision_output.decision_state != "uncertain":
                sess.first_detection_ms = now_ms

            # Sync updated report to DB
            session_manager.sync_report_to_db(sess_id)
            report_snapshot = sess.get_report_schema()

            # Construct latest window evidence DTO
            latest_evidence_dto = None
            evidence_list = sess.decision_engine.get_evidence_list()
            if evidence_list:
                from app.schemas.realtime import WindowEvidenceSchema
                latest_evidence_dto = WindowEvidenceSchema(**evidence_list[-1])

            # Emit clean decision log
            class_label = "SPOOF" if norm_res_val == "SPOOFED" else ("BONAFIDE" if norm_res_val == "REAL" else "UNKNOWN")
            if raw_res_val == "partially_spoofed":
                class_label = "PARTIALLY_SPOOFED"

            VoiceShieldPipelineLogger.decision(
                window_id=win_id,
                risk=report_snapshot.riskLevel,
                result=class_label,
                confidence=conf_val,
                alert=decision_output.notification_required,
            )

            if decision_output.notification_required:
                VoiceShieldPipelineLogger.alert_triggered(
                    call_id=sess_id,
                    window_id=win_id,
                    reason=decision_output.reason,
                )

            # Construct Server Messages
            res_msg = ServerDetectionResultMessage(
                callSessionId=sess_id,
                sequenceNumber=seq_num,
                windowStartMs=win_start_ms,
                windowEndMs=win_end_ms,
                result=str(raw_res_val or norm_res_val).lower(),
                score=decision_output.call_score,
                confidence=decision_output.call_confidence,
                decisionState=decision_output.decision_state,
                userLabel=decision_output.user_label,
                notificationRequired=decision_output.notification_required,
                stateChanged=decision_output.state_changed,
                isEarlyDetection=decision_output.is_early_detection,
                reason=decision_output.reason,
                processingTimeMs=server_total_ms,
                metrics=metrics,
                requestId=resolved_req_id,
                windowId=win_id,
                durationMs=resolved_dur_ms,
                serverTimestampMs=now_ms,
            )

            if ws is not None:
                try:
                    # 1. Emit detection_result
                    await ws.send_text(res_msg.model_dump_json())
                    
                    # 2. Emit call_report_updated
                    report_upd_payload = {
                        "type": "call_report_updated",
                        "callSessionId": sess_id,
                        "report": report_snapshot.model_dump(mode="json"),
                        "latestWindow": latest_evidence_dto.model_dump(mode="json") if latest_evidence_dto else None,
                    }
                    await ws.send_text(json.dumps(report_upd_payload))

                    # 3. If notification is required or risk escalated, emit risk_update
                    if decision_output.notification_required or (decision_output.state_changed and report_snapshot.riskLevel in ("HIGH", "MEDIUM")):
                        risk_upd_payload = {
                            "type": "risk_update",
                            "callSessionId": sess_id,
                            "report": report_snapshot.model_dump(mode="json"),
                        }
                        await ws.send_text(json.dumps(risk_upd_payload))
                except Exception as send_exc:
                    logger.debug("Failed to send detection result over WebSocket (socket closed): %s", send_exc)

    # Shared idempotent call finalization helper
    async def _finalize_call_session(call_session_id: str, ws: WebSocket | None = None) -> None:
        session = session_manager.get_session(call_session_id)
        if session is None:
            return

        async with session.lock:
            if session.status in ("CLOSED", "FINALIZED"):
                return
            session.status = "CLOSED"
            session.closed_at_ms = time.time() * 1000
            session.ended_at = datetime.now(timezone.utc)

        # 1. Await in-flight tasks
        tasks = list(session.active_tasks)
        if tasks:
            try:
                await asyncio.wait(tasks, timeout=5.0)
            except Exception:
                pass
        session.active_tasks.clear()

        # 2. Final report snapshot & logs
        final_report = session.get_report_schema()
        obs = session.decision_engine.observations
        spoof_count = sum(1 for o in obs if o.normalized_result in ("SPOOF", "SPOOFED"))
        partial_count = sum(1 for o in obs if o.raw_result == "partially_spoofed")
        bonafide_count = sum(1 for o in obs if o.normalized_result == "REAL")
        alert_trig = session.decision_engine.has_notified or final_report.riskLevel in ("HIGH", "MEDIUM")

        effective_dur_s = final_report.totalDurationSeconds
        if effective_dur_s > 0:
            VoiceShieldPipelineLogger.tced_call_ended(effective_dur_s)
        VoiceShieldPipelineLogger.call_completed(
            duration_s=effective_dur_s,
            windows=len(obs),
            bonafide=bonafide_count,
            spoofed=spoof_count,
            partially_spoofed=partial_count,
            final_risk=final_report.riskLevel,
            alert_triggered=alert_trig,
            call_id=call_session_id,
        )

        hist_entries = [
            (
                format_window_id(o.sequence_number),
                o.raw_result.upper() if o.raw_result == "partially_spoofed" else ("BONAFIDE" if o.normalized_result == "REAL" else "SPOOF"),
                o.confidence,
            )
            for o in obs
        ]
        VoiceShieldPipelineLogger.call_history(hist_entries, call_id=call_session_id)

        # 5. Persist final report to DB
        session_manager.sync_report_to_db(call_session_id)

        # 6. Send final messages over WebSocket if still open
        if ws is not None:
            try:
                await ws.send_text(
                    json.dumps({
                        "type": "call_report_completed",
                        "callSessionId": call_session_id,
                        "report": final_report.model_dump(mode="json"),
                    })
                )
                await ws.send_text(
                    ServerSessionClosedMessage(callSessionId=call_session_id).model_dump_json()
                )
            except Exception as send_exc:
                logger.debug("Could not send session close frames on WebSocket: %s", send_exc)

    try:
        while True:
            raw_text = await websocket.receive_text()
            receive_time_ms = time.time() * 1000

            try:
                msg_dict = json.loads(raw_text)
            except json.JSONDecodeError:
                await websocket.send_text(
                    ServerErrorMessage(
                        code="INVALID_JSON",
                        message="Incoming payload is not valid JSON",
                    ).model_dump_json()
                )
                continue

            msg_type = msg_dict.get("type")

            # Heartbeat ping/pong
            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time() * 1000}))
                continue

            # 1. Handle call_start
            if msg_type == "call_start":
                try:
                    start_msg = ClientCallStartMessage.model_validate(msg_dict)
                    active_call_session_id = start_msg.callSessionId
                    session = await session_manager.create_session(
                        call_session_id=start_msg.callSessionId,
                        started_at=datetime.now(timezone.utc),
                    )
                    VoiceShieldPipelineLogger.call_started(active_call_session_id)
                    VoiceShieldPipelineLogger.tced_config(
                        new_audio_s=session.tced_manager.config.new_audio_interval_seconds,
                        stride_s=session.tced_manager.config.window_stride_seconds,
                        max_window_s=session.tced_manager.config.max_window_length_seconds,
                    )
                    
                    # 1a. Ack call start
                    await websocket.send_text(
                        json.dumps({
                            "type": "call_started",
                            "callSessionId": active_call_session_id,
                            "timestamp": time.time() * 1000,
                        })
                    )
                    # 1b. Emit initial Call Report
                    report_schema = session.get_report_schema()
                    await websocket.send_text(
                        json.dumps({
                            "type": "call_report_created",
                            "callSessionId": active_call_session_id,
                            "report": report_schema.model_dump(mode="json"),
                        })
                    )
                except ValidationError as ve:
                    await websocket.send_text(
                        ServerErrorMessage(
                            callSessionId=active_call_session_id,
                            code="VALIDATION_ERROR",
                            message=str(ve),
                        ).model_dump_json()
                    )
                continue

            # 2. Handle call_end
            if msg_type == "call_end":
                try:
                    end_msg = ClientCallEndMessage.model_validate(msg_dict)
                    call_session_id = end_msg.callSessionId
                    await _finalize_call_session(call_session_id, ws=websocket)
                    if active_call_session_id == call_session_id:
                        active_call_session_id = None
                    break
                except ValidationError as ve:
                    await websocket.send_text(
                        ServerErrorMessage(
                            callSessionId=active_call_session_id,
                            code="VALIDATION_ERROR",
                            message=str(ve),
                        ).model_dump_json()
                    )
                    continue

            # 3. Handle audio_window
            if msg_type == "audio_window":
                try:
                    window_msg = ClientAudioWindowMessage.model_validate(msg_dict)
                except ValidationError as ve:
                    VoiceShieldPipelineLogger.validation_failed(
                        active_call_session_id or "CALL-UNKNOWN",
                        window_id=1,
                        reason=f"VALIDATION_ERROR: {ve}",
                    )
                    await websocket.send_text(
                        ServerErrorMessage(
                            callSessionId=active_call_session_id,
                            code="VALIDATION_ERROR",
                            message=str(ve),
                        ).model_dump_json()
                    )
                    continue

                call_session_id = window_msg.callSessionId
                req_id = window_msg.requestId or f"REQ_{uuid.uuid4().hex[:8]}"
                calc_dur_ms = window_msg.durationMs if window_msg.durationMs is not None else (window_msg.windowEndMs - window_msg.windowStartMs)

                # Validate audio format compatibility
                if window_msg.audioFormat:
                    if (
                        window_msg.audioFormat.sampleRate != 16000
                        or window_msg.audioFormat.channels != 1
                        or str(window_msg.audioFormat.encoding).lower() not in ("pcm_s16le", "pcm", "s16le")
                    ):
                        VoiceShieldPipelineLogger.validation_failed(
                            call_session_id,
                            window_msg.sequenceNumber,
                            reason=f"UNSUPPORTED_AUDIO_FORMAT: sampleRate={window_msg.audioFormat.sampleRate}, channels={window_msg.audioFormat.channels}, encoding={window_msg.audioFormat.encoding}",
                        )
                        await websocket.send_text(
                            ServerErrorMessage(
                                callSessionId=call_session_id,
                                code="UNSUPPORTED_AUDIO_FORMAT",
                                message=f"Unsupported audio format: sampleRate={window_msg.audioFormat.sampleRate}, channels={window_msg.audioFormat.channels}, encoding={window_msg.audioFormat.encoding}. Expected 16000Hz mono pcm_s16le.",
                            ).model_dump_json()
                        )
                        continue

                # Decode audio data first to check length
                try:
                    raw_audio_bytes = base64.b64decode(window_msg.audio)
                except Exception as exc:
                    VoiceShieldPipelineLogger.validation_failed(call_session_id, window_msg.sequenceNumber, reason=f"MALFORMED_AUDIO: {exc}")
                    await websocket.send_text(
                        ServerErrorMessage(
                            callSessionId=call_session_id,
                            code="MALFORMED_AUDIO",
                            message=f"Failed to base64-decode audio: {exc}",
                        ).model_dump_json()
                    )
                    continue

                if not raw_audio_bytes:
                    VoiceShieldPipelineLogger.validation_failed(call_session_id, window_msg.sequenceNumber, reason="EMPTY_AUDIO")
                    await websocket.send_text(
                        ServerErrorMessage(
                            callSessionId=call_session_id,
                            code="EMPTY_AUDIO",
                            message="Audio payload is empty",
                        ).model_dump_json()
                    )
                    continue

                # Validate active session
                session = session_manager.get_session(call_session_id)
                if session is None or session.status not in ("ACTIVE", "ANALYZING", "RISK_DETECTED", "INTERRUPTED"):
                    await websocket.send_text(
                        ServerErrorMessage(
                            callSessionId=call_session_id,
                            code="SESSION_CLOSED",
                            message="Call session is closed or inactive; dropping window",
                        ).model_dump_json()
                    )
                    continue

                # Deduplicate by sequence number
                if session_manager.is_sequence_processed(call_session_id, window_msg.sequenceNumber):
                    continue
                session_manager.record_sequence(call_session_id, window_msg.sequenceNumber)

                seq_num = window_msg.sequenceNumber
                win_start_ms = window_msg.windowStartMs
                win_end_ms = window_msg.windowEndMs
                win_id = window_msg.windowId or format_window_id(seq_num)

                VoiceShieldPipelineLogger.window_created(
                    window_id=win_id,
                    start_ms=win_start_ms,
                    end_ms=win_end_ms,
                    sequence=seq_num,
                )

                wav_bytes = _pcm_to_wav_bytes(raw_audio_bytes)

                task = asyncio.create_task(
                    _process_and_emit_window(
                        sess_id=call_session_id,
                        seq_num=seq_num,
                        win_id=win_id,
                        win_start_ms=win_start_ms,
                        win_end_ms=win_end_ms,
                        resolved_dur_ms=calc_dur_ms,
                        audio_payload=wav_bytes,
                        server_recv_ms=receive_time_ms,
                        resolved_req_id=req_id,
                        ws=websocket,
                        client_timestamp_ms=window_msg.clientTimestampMs,
                    )
                )
                session_manager.register_task(call_session_id, task)
                continue

            # Diagnostic warning for unhandled message types
            if msg_type not in ("ping", "call_start", "call_end", "audio_window"):
                logger.warning("[WS_WARNING] unhandled_message_type=%s call_id=%s", msg_type, active_call_session_id)

    except WebSocketDisconnect as ws_disc:
        logger.info(
            "[WS_DISCONNECT] call_id=%s reason=client_disconnect code=%s client_initiated=true server_initiated=false",
            active_call_session_id,
            getattr(ws_disc, "code", "1000"),
        )
        if active_call_session_id:
            await _finalize_call_session(active_call_session_id, ws=None)
    except Exception as exc:
        logger.exception(
            "[WS_DISCONNECT] call_id=%s reason=server_exception error=%s client_initiated=false server_initiated=true",
            active_call_session_id,
            exc,
        )
        if active_call_session_id:
            await _finalize_call_session(active_call_session_id, ws=None)
