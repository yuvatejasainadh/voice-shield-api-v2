from __future__ import annotations

import io
import wave
import numpy as np
import soundfile as sf
import av


def generate_wav_bytes(
    duration_seconds: float = 1.0,
    sample_rate: int = 16000,
    frequency_hz: float = 440.0,
    channels: int = 1,
) -> bytes:
    sample_count = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, sample_count, endpoint=False, dtype=np.float32)
    mono = 0.2 * np.sin(2 * np.pi * frequency_hz * t)
    audio = mono if channels == 1 else np.stack([mono] * channels, axis=1)
    pcm = (audio * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def generate_audio_format_bytes(
    fmt: str = "wav",
    duration_seconds: float = 1.0,
    sample_rate: int = 16000,
    frequency_hz: float = 440.0,
) -> bytes:
    """Generate in-memory synthetic audio bytes for any of the 7 supported formats."""
    fmt = fmt.lower()
    if fmt == "wav":
        return generate_wav_bytes(duration_seconds, sample_rate, frequency_hz)

    sample_count = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, sample_count, endpoint=False, dtype=np.float32)
    data = (0.2 * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)

    buf = io.BytesIO()
    if fmt in ("mp3", "ogg", "flac"):
        sf.write(buf, data, sample_rate, format=fmt)
        return buf.getvalue()

    # M4A, AAC, WEBM via PyAV
    format_map = {
        "m4a": ("ipod", "aac"),
        "aac": ("adts", "aac"),
        "webm": ("webm", "opus"),
    }
    format_name, codec_name = format_map.get(fmt, ("wav", "pcm_s16le"))

    container = av.open(buf, mode="w", format=format_name)
    stream = container.add_stream(codec_name, rate=sample_rate)
    stream.layout = "mono"
    stream.format = "flt"

    frame_size = 1024
    for i in range(0, len(data), frame_size):
        chunk = data[i : i + frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="flt", layout="mono")
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    return buf.getvalue()
