"""Gradio demo interface for ASR Server."""

import gradio as gr
import requests
import websockets
import asyncio
import json
import numpy as np
import io
import wave
import subprocess
import os
import time
import tempfile
from dataclasses import dataclass, field
from typing import Optional

# Server configuration (must match config.yaml server.port)
API_BASE_URL = "http://localhost:8005"
WS_URL = "ws://localhost:8005/v1/audio/stream"

# Common language codes for Whisper
LANGUAGES = [
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Dutch", "nl"),
    ("Polish", "pl"),
    ("Russian", "ru"),
    ("Japanese", "ja"),
    ("Chinese", "zh"),
    ("Korean", "ko"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Auto-detect", "auto"),
]


@dataclass
class StreamingSession:
    """Manages WebSocket connection for streaming transcription."""
    ws: Optional[object] = None  # WebSocket connection (avoiding deprecated type hint)
    connected: bool = False
    latest_text: str = ""
    audio_buffer: bytearray = field(default_factory=bytearray)
    start_time: float = 0.0
    stop_time: float = 0.0


# Global streaming session (persists across .stream() calls)
_streaming_session: Optional[StreamingSession] = None

# File-based audio accumulator (globals don't persist across Gradio worker processes)
_LIVE_AUDIO_FILE = os.path.join(tempfile.gettempdir(), "asr_live_audio_buffer.pcm")
_LIVE_TEXT_FILE = os.path.join(tempfile.gettempdir(), "asr_live_latest_text.txt")
_FRESH_RECORDING_MARKER = os.path.join(tempfile.gettempdir(), "asr_fresh_recording.marker")
_LAST_RESULT_FILE = os.path.join(tempfile.gettempdir(), "asr_last_result.json")  # Cache result for duplicate calls

# VAD Live Transcription state files
_VAD_STATE_FILE = os.path.join(tempfile.gettempdir(), "asr_vad_state.json")
_VAD_AUDIO_FILE = os.path.join(tempfile.gettempdir(), "asr_vad_audio.pcm")
_VAD_TRANSCRIPT_FILE = os.path.join(tempfile.gettempdir(), "asr_vad_transcript.txt")


def _convert_audio_chunk(audio_tuple) -> bytes:
    """Convert Gradio audio tuple to 16kHz mono PCM bytes.

    Preserves original amplitude - does NOT normalize each chunk independently.
    Input is typically int16 from Gradio's audio component.
    """
    if audio_tuple is None:
        return b""

    sample_rate, audio_data = audio_tuple

    # Work in float32 for processing, preserving int16 scale [-32768, 32767]
    audio_data = audio_data.astype(np.float32)

    # Convert to mono if stereo (average channels)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        # Calculate exact output length to preserve duration
        num_samples = int(len(audio_data) * 16000 / sample_rate)
        # Use endpoint=False for correct resampling - avoids extending past valid indices
        # and ensures proper sample alignment at chunk boundaries
        audio_data = np.interp(
            np.linspace(0, len(audio_data), num_samples, endpoint=False),
            np.arange(len(audio_data)),
            audio_data
        )

    # Convert back to int16, clipping to valid range (no per-chunk normalization!)
    audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)

    return audio_data.tobytes()


def _create_wav_bytes(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """Convert PCM bytes to WAV format."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)
    return buffer.getvalue()


def _calculate_rms(audio_data: np.ndarray) -> float:
    """Calculate RMS energy of audio chunk (0.0 to 1.0 scale).

    Used for Voice Activity Detection to identify speech vs silence.
    """
    if audio_data is None or len(audio_data) == 0:
        return 0.0
    audio_float = audio_data.astype(np.float32)
    if len(audio_float.shape) > 1:
        audio_float = audio_float.mean(axis=1)
    # Normalize to -1 to 1 range for consistent RMS
    max_val = max(np.abs(audio_float).max(), 1e-7)
    audio_float = audio_float / max_val
    return float(np.sqrt(np.mean(audio_float ** 2)))


# =============================================================================
# VAD Live Transcription State Management
# =============================================================================

def _get_vad_state() -> dict:
    """Load VAD state from file."""
    if os.path.exists(_VAD_STATE_FILE):
        try:
            with open(_VAD_STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "is_speaking": False,
        "silence_ms": 0,
        "speech_ms": 0,
        "last_chunk_time": 0.0,
        # Silence duration statistics
        "silence_sum": 0.0,
        "silence_count": 0,
        "silence_peak": 0.0,
        "silence_min": 999999.0,
        # RMS statistics - overall
        "rms_sum": 0.0,
        "rms_count": 0,
        "rms_peak": 0.0,
        "rms_min": 1.0,
        # RMS statistics - during speech
        "rms_speech_sum": 0.0,
        "rms_speech_count": 0,
        "rms_speech_peak": 0.0,
        "rms_speech_min": 1.0,
        # RMS statistics - during silence
        "rms_silence_sum": 0.0,
        "rms_silence_count": 0,
        "rms_silence_peak": 0.0,
        "rms_silence_min": 1.0,
    }


def _set_vad_state(state: dict):
    """Save VAD state to file."""
    with open(_VAD_STATE_FILE, "w") as f:
        json.dump(state, f)


def _get_vad_transcript() -> str:
    """Read accumulated transcript from file."""
    if os.path.exists(_VAD_TRANSCRIPT_FILE):
        with open(_VAD_TRANSCRIPT_FILE, "r") as f:
            return f.read()
    return ""


def _append_vad_transcript(text: str):
    """Append new transcription with timestamp to transcript file."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {text}\n"
    with open(_VAD_TRANSCRIPT_FILE, "a") as f:
        f.write(entry)


def _clear_vad_files():
    """Clear all VAD state and audio files."""
    for f in [_VAD_STATE_FILE, _VAD_AUDIO_FILE, _VAD_TRANSCRIPT_FILE]:
        if os.path.exists(f):
            os.remove(f)


def check_server_health():
    """Check if the ASR server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def load_server_config():
    """Load current config from server."""
    try:
        response = requests.get(f"{API_BASE_URL}/v1/config", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


def get_config_display():
    """Get formatted config for display."""
    config = load_server_config()
    if config is None:
        return "Server not available", "en", 4, False, 0.5, 250, 700

    whisper = config.get("whisper", {})
    vad = config.get("vad", {})

    status = f"""**Server Status:** Connected

**Whisper Settings:**
- Model: `{whisper.get('model_path', 'unknown')}`
- Language: `{whisper.get('language', 'en')}`
- Threads: `{whisper.get('threads', 4)}`

**VAD Settings:**
- Enabled: `{vad.get('enabled', False)}`
- Threshold: `{vad.get('threshold', 0.5)}`
- Min Speech Duration: `{vad.get('min_speech_duration_ms', 250)}ms`
- Min Silence Duration: `{vad.get('min_silence_duration_ms', 700)}ms`

**Hardware:** GPU (Metal) enabled by default
"""

    return (
        status,
        whisper.get('language', 'en'),
        whisper.get('threads', 4),
        vad.get('enabled', False),
        vad.get('threshold', 0.5),
        vad.get('min_speech_duration_ms', 250),
        vad.get('min_silence_duration_ms', 700)
    )


def save_config(language, threads, vad_enabled, vad_threshold, min_speech_ms, min_silence_ms):
    """Save config to server."""
    if not check_server_health():
        return "Error: Server not available"

    try:
        payload = {
            "whisper_language": language,
            "whisper_threads": int(threads),
            "vad_enabled": vad_enabled,
            "vad_threshold": float(vad_threshold),
            "vad_min_speech_duration_ms": int(min_speech_ms),
            "vad_min_silence_duration_ms": int(min_silence_ms)
        }

        response = requests.post(
            f"{API_BASE_URL}/v1/config",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            changes = result.get("changes", [])
            return f"Config saved successfully.\nChanges: {', '.join(changes)}"
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


def restart_server():
    """Restart the ASR server using asr.sh."""
    try:
        # Find the project root (where asr.sh is)
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        asr_script = os.path.join(script_dir, "asr.sh")

        if not os.path.exists(asr_script):
            return "Error: asr.sh not found"

        # Restart just the API server
        result = subprocess.run(
            ["bash", asr_script, "restart"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return "Server restarted successfully.\nNote: Gradio UI reconnected."
        else:
            return f"Restart failed:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Restart timed out"
    except Exception as e:
        return f"Error: {str(e)}"


def transcribe_file(audio_file, language_override=None):
    """Transcribe uploaded audio file via HTTP API."""
    if audio_file is None:
        return "No audio file provided."

    if not check_server_health():
        return "Error: ASR Server is not running. Start it with: python src/server.py"

    try:
        with open(audio_file, "rb") as f:
            files = {"file": (audio_file, f, "audio/wav")}
            data = {}
            if language_override and language_override != "default":
                data["language"] = language_override

            response = requests.post(
                f"{API_BASE_URL}/v1/audio/transcriptions",
                files=files,
                data=data,
                timeout=60
            )

        if response.status_code == 200:
            return response.json().get("text", "")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


def transcribe_recording_batch(audio, language_override=None):
    """Transcribe recorded audio in batch mode via HTTP API."""
    if audio is None:
        return "No audio recorded."

    if not check_server_health():
        return "Error: ASR Server is not running. Start it with: python src/server.py"

    try:
        sample_rate, audio_data = audio

        # Convert to 16kHz mono if needed
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), num_samples),
                np.arange(len(audio_data)),
                audio_data
            )
            sample_rate = 16000

        # Normalize and convert to 16-bit PCM
        audio_data = audio_data.astype(np.float32)
        audio_data = audio_data / max(np.abs(audio_data).max(), 1e-7)
        audio_data = (audio_data * 32767).astype(np.int16)

        # Create WAV in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(audio_data.tobytes())

        buffer.seek(0)

        # Send to API
        files = {"file": ("recording.wav", buffer, "audio/wav")}
        data = {}
        if language_override and language_override != "default":
            data["language"] = language_override

        response = requests.post(
            f"{API_BASE_URL}/v1/audio/transcriptions",
            files=files,
            data=data,
            timeout=60
        )

        if response.status_code == 200:
            return response.json().get("text", "")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


async def transcribe_streaming_async(audio, state):
    """Stream audio to WebSocket and get transcriptions."""
    if audio is None:
        return "No audio recorded.", state

    if not check_server_health():
        return "Error: ASR Server is not running. Start it with: python src/server.py", state

    try:
        sample_rate, audio_data = audio

        # Convert to 16kHz mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        if sample_rate != 16000:
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), num_samples),
                np.arange(len(audio_data)),
                audio_data
            )

        # Normalize and convert to 16-bit PCM
        audio_data = audio_data.astype(np.float32)
        audio_data = audio_data / max(np.abs(audio_data).max(), 1e-7)
        pcm_data = (audio_data * 32767).astype(np.int16).tobytes()

        latest_transcription = ""

        async with websockets.connect(WS_URL) as ws:
            # Send audio in chunks
            chunk_size = 16000  # 500ms at 16kHz, 16-bit (16000 samples = 32000 bytes / 2)
            for i in range(0, len(pcm_data), chunk_size * 2):
                chunk = pcm_data[i:i + chunk_size * 2]
                await ws.send(chunk)

                # Check for responses (non-blocking)
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    msg = json.loads(response)
                    # Server sends full transcription each time, just keep latest
                    if msg.get("text"):
                        latest_transcription = msg["text"]
                except asyncio.TimeoutError:
                    pass

            # Signal end of stream
            await ws.send("end")

            # Drain all remaining messages until we get the "final" one
            # (there may be queued partial messages)
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=10)
                    msg = json.loads(response)
                    if msg.get("text"):
                        latest_transcription = msg["text"]
                    # Stop when we get the final message
                    if msg.get("type") == "final" or msg.get("is_utterance_complete"):
                        break
            except asyncio.TimeoutError:
                pass
            except websockets.exceptions.ConnectionClosed:
                pass

        return latest_transcription if latest_transcription else "No speech detected.", state

    except Exception as e:
        return f"Error: {str(e)}", state


def transcribe_streaming(audio, state):
    """Wrapper to run async streaming in sync context."""
    return asyncio.run(transcribe_streaming_async(audio, state))


def transcribe_recording(audio, mode, language_override, state):
    """Route to batch or streaming based on mode."""
    if mode == "Batch":
        return transcribe_recording_batch(audio, language_override), state
    else:
        return transcribe_streaming(audio, state)


# =============================================================================
# Live Recording with Side-by-Side Comparison
# =============================================================================

async def _connect_streaming_session():
    """Connect to WebSocket for streaming transcription."""
    global _streaming_session
    _streaming_session = StreamingSession()
    _streaming_session.ws = await websockets.connect(WS_URL)
    _streaming_session.connected = True
    _streaming_session.start_time = time.time()
    _streaming_session.audio_buffer = bytearray()
    _streaming_session.latest_text = ""


async def _send_chunk_to_ws(pcm_data: bytes):
    """Send audio chunk to WebSocket and poll for responses."""
    global _streaming_session
    if _streaming_session is None or not _streaming_session.connected:
        return ""

    try:
        await _streaming_session.ws.send(pcm_data)

        # Non-blocking poll for transcription updates
        try:
            response = await asyncio.wait_for(
                _streaming_session.ws.recv(),
                timeout=0.05
            )
            msg = json.loads(response)
            if msg.get("text"):
                _streaming_session.latest_text = msg["text"]
        except asyncio.TimeoutError:
            pass

        return _streaming_session.latest_text
    except Exception:
        return _streaming_session.latest_text if _streaming_session else ""


async def _finalize_streaming():
    """Finalize streaming session and return final transcription with latency."""
    global _streaming_session
    if _streaming_session is None or not _streaming_session.connected:
        return "", 0.0

    try:
        _streaming_session.stop_time = time.time()
        await _streaming_session.ws.send("end")

        # Drain all remaining messages until final
        while True:
            try:
                response = await asyncio.wait_for(
                    _streaming_session.ws.recv(),
                    timeout=10
                )
                msg = json.loads(response)
                if msg.get("text"):
                    _streaming_session.latest_text = msg["text"]
                if msg.get("type") == "final" or msg.get("is_utterance_complete"):
                    break
            except asyncio.TimeoutError:
                break
            except websockets.exceptions.ConnectionClosed:
                break

        latency = time.time() - _streaming_session.stop_time
        final_text = _streaming_session.latest_text or "No speech detected."

        await _streaming_session.ws.close()
        _streaming_session.connected = False

        return final_text, latency

    except Exception as e:
        return f"Error: {str(e)}", 0.0


def _get_live_text() -> str:
    """Read latest transcription text from file."""
    if os.path.exists(_LIVE_TEXT_FILE):
        with open(_LIVE_TEXT_FILE, "r") as f:
            return f.read()
    return ""


def _set_live_text(text: str):
    """Write latest transcription text to file."""
    with open(_LIVE_TEXT_FILE, "w") as f:
        f.write(text)


def handle_stream_chunk(audio_chunk):
    """
    Handle incoming audio chunk during live recording.
    Called by Gradio's .stream() event every 0.5 seconds.

    NOTE: With streaming=True, Gradio sends INCREMENTAL audio chunks,
    not cumulative. We append each chunk to build the full recording.

    Returns: streaming_text
    """
    import sys
    print(f"[DEBUG] handle_stream_chunk CALLED, audio_chunk type={type(audio_chunk)}", file=sys.stderr, flush=True)

    if audio_chunk is None:
        print("[DEBUG] audio_chunk is None", file=sys.stderr, flush=True)
        return _get_live_text() or "Listening..."

    # Log audio chunk details
    sample_rate, audio_data = audio_chunk
    print(f"[DEBUG] audio_chunk: sample_rate={sample_rate}, shape={audio_data.shape}, dtype={audio_data.dtype}", file=sys.stderr, flush=True)

    if not check_server_health():
        print("[DEBUG] Server not healthy", file=sys.stderr, flush=True)
        return "Error: Server not available"

    # Check for fresh recording marker - if present, clear old files
    if os.path.exists(_FRESH_RECORDING_MARKER):
        print("[DEBUG] Fresh recording marker found - clearing old files", file=sys.stderr, flush=True)
        os.remove(_FRESH_RECORDING_MARKER)
        if os.path.exists(_LIVE_AUDIO_FILE):
            os.remove(_LIVE_AUDIO_FILE)
        if os.path.exists(_LIVE_TEXT_FILE):
            os.remove(_LIVE_TEXT_FILE)

    # Convert incremental audio chunk to PCM and APPEND to file
    # Gradio streaming sends small incremental chunks (~100ms each)
    pcm_data = _convert_audio_chunk(audio_chunk)
    print(f"[DEBUG] Converted PCM: {len(pcm_data)} bytes", file=sys.stderr, flush=True)

    with open(_LIVE_AUDIO_FILE, "ab") as f:  # append mode
        f.write(pcm_data)

    print(f"[DEBUG] Wrote to {_LIVE_AUDIO_FILE}", file=sys.stderr, flush=True)

    # Get accumulated file size for transcription decision
    buffer_size = os.path.getsize(_LIVE_AUDIO_FILE)
    buffer_duration_ms = buffer_size / 32  # 16kHz * 2 bytes / 1000
    print(f"[DEBUG] handle_stream_chunk: chunk={len(pcm_data)}, total={buffer_size} bytes, duration={buffer_duration_ms:.0f}ms", file=sys.stderr, flush=True)

    # Transcribe every ~2 seconds of new audio using HTTP batch API (simpler than WebSocket)
    # Only transcribe if we have at least 2 seconds and haven't transcribed recently
    if buffer_duration_ms >= 2000:
        try:
            # Read accumulated audio and convert to WAV
            with open(_LIVE_AUDIO_FILE, "rb") as f:
                pcm_data = f.read()
            wav_data = _create_wav_bytes(pcm_data, sample_rate=16000)

            # Use batch API for simplicity (it's fast enough for live preview)
            files = {"file": ("chunk.wav", io.BytesIO(wav_data), "audio/wav")}
            response = requests.post(
                f"{API_BASE_URL}/v1/audio/transcriptions",
                files=files,
                timeout=5
            )
            if response.status_code == 200:
                text = response.json().get("text", "")
                if text:
                    _set_live_text(text)
                    print(f"[DEBUG] Live transcription: {text[:50]}...")
        except Exception as e:
            print(f"[DEBUG] Live transcription error: {e}")

    return _get_live_text() or "Listening..."


def handle_recording_stop(audio_data, language_override):
    """
    Handle recording stop - transcribe the complete audio.

    With streaming=False, we get the complete, gap-free audio directly from the component.

    Returns: (transcription_text, latency, playback_audio)
    """
    import sys
    print(f"[DEBUG] handle_recording_stop CALLED, audio_data type={type(audio_data)}", file=sys.stderr, flush=True)

    if audio_data is None:
        print("[DEBUG] No audio data received", file=sys.stderr, flush=True)
        return "No audio recorded.", "Latency: --", None

    sample_rate, audio_array = audio_data
    print(f"[DEBUG] Audio: sr={sample_rate}, shape={audio_array.shape}, dtype={audio_array.dtype}", file=sys.stderr, flush=True)

    # Convert to 16kHz mono PCM
    audio_float = audio_array.astype(np.float32)
    if len(audio_float.shape) > 1:
        audio_float = audio_float.mean(axis=1)
    if sample_rate != 16000:
        num_samples = int(len(audio_float) * 16000 / sample_rate)
        audio_float = np.interp(
            np.linspace(0, len(audio_float), num_samples, endpoint=False),
            np.arange(len(audio_float)),
            audio_float
        )
    audio_int16 = np.clip(audio_float, -32768, 32767).astype(np.int16)
    pcm_buffer = audio_int16.tobytes()

    duration_sec = len(audio_int16) / 16000
    print(f"[DEBUG] Converted: {len(pcm_buffer)} bytes, {duration_sec:.2f}s duration", file=sys.stderr, flush=True)

    if not check_server_health():
        return "Error: Server not available", "Latency: --", None

    # Convert PCM to WAV
    wav_data = _create_wav_bytes(pcm_buffer, sample_rate=16000)

    # Batch transcription via HTTP
    batch_start = time.time()
    try:
        files = {"file": ("recording.wav", io.BytesIO(wav_data), "audio/wav")}
        data = {}
        if language_override and language_override != "default":
            data["language"] = language_override

        response = requests.post(
            f"{API_BASE_URL}/v1/audio/transcriptions",
            files=files,
            data=data,
            timeout=60
        )

        if response.status_code == 200:
            batch_text = response.json().get("text", "") or "No speech detected."
        else:
            batch_text = f"Error: {response.status_code}"
    except Exception as e:
        batch_text = f"Batch error: {str(e)}"
    batch_latency = time.time() - batch_start
    batch_lat_str = f"Latency: {batch_latency:.2f}s"

    # Create playback audio tuple (sample_rate, numpy array)
    pcm_array = np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32767.0
    playback_audio = (16000, pcm_array)

    return batch_text, batch_lat_str, playback_audio


def reset_live_outputs():
    """Reset outputs when starting a new recording."""
    return "", "Latency: --", None


# =============================================================================
# VAD Live Transcription Handlers
# =============================================================================

# VAD tuning constants
VAD_MIN_SPEECH_MS = 300       # Minimum speech duration to consider valid utterance
VAD_CHUNK_DURATION_MS = 100   # Approximate chunk duration from Gradio streaming


def _format_vad_status(state: dict, rms: float, silence_ms: float, is_speaking: bool) -> str:
    """Format VAD statistics as a stable markdown table."""
    # Calculate averages - overall RMS
    rms_avg = state["rms_sum"] / state["rms_count"] if state["rms_count"] > 0 else 0.0
    rms_min = state["rms_min"] if state["rms_min"] < 1.0 else 0.0

    # Calculate averages - speech RMS
    rms_speech_avg = state["rms_speech_sum"] / state["rms_speech_count"] if state["rms_speech_count"] > 0 else 0.0
    rms_speech_min = state["rms_speech_min"] if state["rms_speech_min"] < 1.0 else 0.0

    # Calculate averages - silence RMS
    rms_silence_avg = state["rms_silence_sum"] / state["rms_silence_count"] if state["rms_silence_count"] > 0 else 0.0
    rms_silence_min = state["rms_silence_min"] if state["rms_silence_min"] < 1.0 else 0.0

    # Silence duration stats
    silence_avg = state["silence_sum"] / state["silence_count"] if state["silence_count"] > 0 else 0.0
    silence_min = state["silence_min"] if state["silence_min"] < 999999.0 else 0.0

    speaking_str = "**Yes**" if is_speaking else "No"

    return f"""| Metric | Current | Avg | Peak | Min |
|--------|---------|-----|------|-----|
| Speaking | {speaking_str} | - | - | - |
| Silence | {silence_ms:.0f}ms | {silence_avg:.0f}ms | {state['silence_peak']:.0f}ms | {silence_min:.0f}ms |
| RMS (Overall) | {rms:.3f} | {rms_avg:.3f} | {state['rms_peak']:.3f} | {rms_min:.3f} |
| RMS (Speech) | - | {rms_speech_avg:.3f} | {state['rms_speech_peak']:.3f} | {rms_speech_min:.3f} |
| RMS (Silence) | - | {rms_silence_avg:.3f} | {state['rms_silence_peak']:.3f} | {rms_silence_min:.3f} |"""


def handle_vad_stream_chunk(audio_chunk, silence_threshold, min_silence_ms, language_override):
    """
    Process audio chunk with VAD logic for automatic utterance detection.

    When silence is detected after speech, transcribes the accumulated audio
    and appends the result to the rolling transcript.

    Returns: (status_text, transcript_text)
    """
    import sys

    if audio_chunk is None:
        transcript = _get_vad_transcript()
        state = _get_vad_state()
        status = _format_vad_status(state, 0.0, 0.0, False)
        return status, transcript or "Transcriptions will appear here..."

    if not check_server_health():
        return "**Server not available**", _get_vad_transcript()

    sample_rate, audio_data = audio_chunk

    # Calculate RMS energy for this chunk
    rms = _calculate_rms(audio_data)
    is_speech = rms > silence_threshold

    # Convert chunk to PCM and append to audio buffer
    pcm_data = _convert_audio_chunk(audio_chunk)
    with open(_VAD_AUDIO_FILE, "ab") as f:
        f.write(pcm_data)

    # Load and update VAD state
    state = _get_vad_state()
    current_time = time.time()

    # Update RMS statistics - overall (always, for every chunk)
    state["rms_sum"] = state.get("rms_sum", 0.0) + rms
    state["rms_count"] = state.get("rms_count", 0) + 1
    state["rms_peak"] = max(state.get("rms_peak", 0.0), rms)
    state["rms_min"] = min(state.get("rms_min", 1.0), rms)

    # Update RMS statistics - speech vs silence (based on threshold)
    if is_speech:
        state["rms_speech_sum"] = state.get("rms_speech_sum", 0.0) + rms
        state["rms_speech_count"] = state.get("rms_speech_count", 0) + 1
        state["rms_speech_peak"] = max(state.get("rms_speech_peak", 0.0), rms)
        state["rms_speech_min"] = min(state.get("rms_speech_min", 1.0), rms)
    else:
        state["rms_silence_sum"] = state.get("rms_silence_sum", 0.0) + rms
        state["rms_silence_count"] = state.get("rms_silence_count", 0) + 1
        state["rms_silence_peak"] = max(state.get("rms_silence_peak", 0.0), rms)
        state["rms_silence_min"] = min(state.get("rms_silence_min", 1.0), rms)

    # Estimate chunk duration from audio length
    chunk_ms = len(audio_data) / sample_rate * 1000

    if is_speech:
        # Speech detected - reset silence counter, accumulate speech duration
        state["silence_ms"] = 0
        state["speech_ms"] = state.get("speech_ms", 0) + chunk_ms
        if not state["is_speaking"]:
            state["is_speaking"] = True
            print(f"[VAD] Speech started (RMS={rms:.3f})", file=sys.stderr, flush=True)
    else:
        # Silence detected - accumulate silence duration
        state["silence_ms"] = state.get("silence_ms", 0) + chunk_ms

        # Update silence statistics (only when in silence)
        current_silence = state["silence_ms"]
        state["silence_sum"] = state.get("silence_sum", 0.0) + chunk_ms
        state["silence_count"] = state.get("silence_count", 0) + 1
        state["silence_peak"] = max(state.get("silence_peak", 0.0), current_silence)
        if current_silence > 0:
            state["silence_min"] = min(state.get("silence_min", 999999.0), current_silence)

        if state["is_speaking"] and state["speech_ms"] >= VAD_MIN_SPEECH_MS:
            # We had valid speech - check if silence is long enough to trigger transcription
            if state["silence_ms"] >= min_silence_ms:
                print(f"[VAD] Utterance complete. Speech={state['speech_ms']:.0f}ms, Silence={state['silence_ms']:.0f}ms", file=sys.stderr, flush=True)

                # Read and immediately clear buffer BEFORE transcription
                # This allows new audio to accumulate while we transcribe
                audio_buffer = b""
                if os.path.exists(_VAD_AUDIO_FILE):
                    with open(_VAD_AUDIO_FILE, "rb") as f:
                        audio_buffer = f.read()
                    # Clear buffer immediately so new audio starts fresh
                    os.remove(_VAD_AUDIO_FILE)

                # Reset utterance state (but keep session statistics)
                state["is_speaking"] = False
                state["speech_ms"] = 0
                state["silence_ms"] = 0
                _set_vad_state(state)

                # Now transcribe (new audio accumulates in fresh buffer during this call)
                if len(audio_buffer) > 0:
                    wav_data = _create_wav_bytes(audio_buffer, sample_rate=16000)

                    try:
                        files = {"file": ("utterance.wav", io.BytesIO(wav_data), "audio/wav")}
                        data = {}
                        if language_override and language_override != "default":
                            data["language"] = language_override

                        response = requests.post(
                            f"{API_BASE_URL}/v1/audio/transcriptions",
                            files=files,
                            data=data,
                            timeout=30
                        )

                        if response.status_code == 200:
                            text = response.json().get("text", "").strip()
                            if text:
                                _append_vad_transcript(text)
                                print(f"[VAD] Transcribed: {text[:50]}...", file=sys.stderr, flush=True)
                    except Exception as e:
                        print(f"[VAD] Transcription error: {e}", file=sys.stderr, flush=True)
        else:
            # Either no speech yet or speech too short
            if state["is_speaking"] and state["silence_ms"] >= min_silence_ms:
                # Reset if silence is long enough
                state["is_speaking"] = False
                state["speech_ms"] = 0
                state["silence_ms"] = 0
                if os.path.exists(_VAD_AUDIO_FILE):
                    os.remove(_VAD_AUDIO_FILE)

    state["last_chunk_time"] = current_time
    _set_vad_state(state)

    # Build stable status display
    status = _format_vad_status(state, rms, state["silence_ms"], state["is_speaking"])

    transcript = _get_vad_transcript()
    return status, transcript or "Transcriptions will appear here..."


def clear_vad_state():
    """Clear VAD transcript and reset state."""
    _clear_vad_files()
    initial_status = """| Metric | Current | Avg | Peak | Min |
|--------|---------|-----|------|-----|
| Speaking | No | - | - | - |
| Silence | 0ms | 0ms | 0ms | 0ms |
| RMS (Overall) | 0.000 | 0.000 | 0.000 | 0.000 |
| RMS (Speech) | - | 0.000 | 0.000 | 0.000 |
| RMS (Silence) | - | 0.000 | 0.000 | 0.000 |"""
    return "Transcriptions will appear here...", initial_status


# Build the Gradio interface
with gr.Blocks(title="ASR Server Demo") as demo:
    gr.Markdown(
        f"""
        # ASR Server Demo
        Test speech recognition with file uploads or live recording.
        [API Documentation]({API_BASE_URL}/docs) | [Health Check]({API_BASE_URL}/health)
        """
    )

    with gr.Tabs():
        # File Upload Tab
        with gr.TabItem("File Upload"):
            with gr.Row():
                file_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    sources=["upload"]
                )
            with gr.Row():
                file_lang_override = gr.Dropdown(
                    choices=[("Use server default", "default")] + LANGUAGES,
                    value="default",
                    label="Language (per-request override)",
                    scale=1
                )
            file_transcribe_btn = gr.Button("Transcribe", variant="primary")
            file_output = gr.Textbox(
                label="Transcription",
                lines=5,
                placeholder="Transcription will appear here..."
            )

            file_transcribe_btn.click(
                fn=transcribe_file,
                inputs=[file_input, file_lang_override],
                outputs=[file_output]
            )

        # Live Recording Tab - Side-by-Side Comparison
        with gr.TabItem("Live Recording"):
            gr.Markdown(
                """
                **Record from microphone** and get transcription when recording stops.
                """
            )

            # Audio input
            recording_input = gr.Audio(
                label="Record Audio",
                type="numpy",
                sources=["microphone"],
                streaming=False
            )

            with gr.Row():
                rec_lang_override = gr.Dropdown(
                    choices=[("Use server default", "default")] + LANGUAGES,
                    value="default",
                    label="Language",
                    scale=1
                )

            # Transcription output
            transcription_output = gr.Textbox(
                label="Transcription",
                lines=5,
                placeholder="Transcription appears after recording stops...",
                interactive=False
            )
            transcription_latency = gr.Markdown("Latency: --")

            # Playback component for final recording
            playback_audio = gr.Audio(
                label="Recorded Audio (playback)",
                type="numpy",
                interactive=False
            )

            # Stop recording event - run transcription and show playback
            recording_input.stop_recording(
                fn=handle_recording_stop,
                inputs=[recording_input, rec_lang_override],
                outputs=[transcription_output, transcription_latency, playback_audio]
            )

            # Start recording event - reset outputs
            recording_input.start_recording(
                fn=reset_live_outputs,
                outputs=[transcription_output, transcription_latency, playback_audio]
            )

        # VAD Live Transcription Tab
        with gr.TabItem("VAD Live"):
            gr.Markdown(
                """
                **Voice Activity Detection (VAD) Transcription**

                Speak naturally - transcription happens automatically when you pause.
                """
            )

            vad_audio_input = gr.Audio(
                label="Microphone (click to start)",
                type="numpy",
                sources=["microphone"],
                streaming=True
            )

            vad_status = gr.Markdown("""| Metric | Current | Avg | Peak | Min |
|--------|---------|-----|------|-----|
| Speaking | No | - | - | - |
| Silence | 0ms | 0ms | 0ms | 0ms |
| RMS (Overall) | 0.000 | 0.000 | 0.000 | 0.000 |
| RMS (Speech) | - | 0.000 | 0.000 | 0.000 |
| RMS (Silence) | - | 0.000 | 0.000 | 0.000 |""")

            vad_transcript = gr.Textbox(
                label="Transcript",
                lines=10,
                placeholder="Transcriptions will appear here with timestamps...",
                interactive=False
            )

            vad_clear_btn = gr.Button("Clear Transcript", variant="secondary")

            with gr.Accordion("VAD Settings", open=False):
                with gr.Row():
                    vad_silence_threshold = gr.Slider(
                        minimum=0.01,
                        maximum=0.5,
                        step=0.01,
                        value=0.25,
                        label="Silence Threshold (RMS)",
                        info="Lower = more sensitive to quiet speech"
                    )
                    vad_min_silence = gr.Slider(
                        minimum=300,
                        maximum=2000,
                        step=100,
                        value=700,
                        label="Min Silence Duration (ms)",
                        info="How long to wait before triggering transcription"
                    )
                with gr.Row():
                    vad_lang_override = gr.Dropdown(
                        choices=[("Use server default", "default")] + LANGUAGES,
                        value="default",
                        label="Language"
                    )

            # Streaming audio triggers VAD processing
            vad_audio_input.stream(
                fn=handle_vad_stream_chunk,
                inputs=[vad_audio_input, vad_silence_threshold, vad_min_silence, vad_lang_override],
                outputs=[vad_status, vad_transcript]
            )

            # Clear button resets transcript and state
            vad_clear_btn.click(
                fn=clear_vad_state,
                outputs=[vad_transcript, vad_status]
            )

        # Settings Tab
        with gr.TabItem("Settings"):
            gr.Markdown("### Server Configuration")

            with gr.Row():
                with gr.Column(scale=1):
                    config_status = gr.Markdown("Loading...")
                    refresh_btn = gr.Button("Refresh Status", variant="secondary")

            gr.Markdown("### Edit Configuration")
            gr.Markdown("*Changes are applied immediately and saved to config.yaml*")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Whisper Settings**")
                    settings_language = gr.Dropdown(
                        choices=LANGUAGES,
                        value="en",
                        label="Default Language"
                    )
                    settings_threads = gr.Slider(
                        minimum=1,
                        maximum=16,
                        step=1,
                        value=4,
                        label="CPU Threads"
                    )

                with gr.Column():
                    gr.Markdown("**VAD Settings**")
                    settings_vad_enabled = gr.Checkbox(
                        value=False,
                        label="Enable Voice Activity Detection"
                    )
                    settings_vad_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.5,
                        label="VAD Threshold (sensitivity)"
                    )
                    settings_min_speech = gr.Slider(
                        minimum=50,
                        maximum=1000,
                        step=50,
                        value=250,
                        label="Min Speech Duration (ms)"
                    )
                    settings_min_silence = gr.Slider(
                        minimum=100,
                        maximum=2000,
                        step=100,
                        value=700,
                        label="Min Silence Duration (ms)"
                    )

            with gr.Row():
                save_config_btn = gr.Button("Save Configuration", variant="primary")
                restart_btn = gr.Button("Restart Server", variant="secondary")

            config_result = gr.Textbox(
                label="Result",
                lines=2,
                placeholder="Status messages will appear here..."
            )

            # Load initial config
            def load_config_to_ui():
                return get_config_display()

            refresh_btn.click(
                fn=load_config_to_ui,
                outputs=[
                    config_status,
                    settings_language,
                    settings_threads,
                    settings_vad_enabled,
                    settings_vad_threshold,
                    settings_min_speech,
                    settings_min_silence
                ]
            )

            save_config_btn.click(
                fn=save_config,
                inputs=[
                    settings_language,
                    settings_threads,
                    settings_vad_enabled,
                    settings_vad_threshold,
                    settings_min_speech,
                    settings_min_silence
                ],
                outputs=[config_result]
            )

            restart_btn.click(
                fn=restart_server,
                outputs=[config_result]
            )

            # Auto-load config on tab load
            demo.load(
                fn=load_config_to_ui,
                outputs=[
                    config_status,
                    settings_language,
                    settings_threads,
                    settings_vad_enabled,
                    settings_vad_threshold,
                    settings_min_speech,
                    settings_min_silence
                ]
            )

    # Status bar
    gr.Markdown(
        f"""
        ---
        **Note:** Ensure the ASR server is running on `localhost:8005` before testing.
        """
    )


def main():
    """Launch the Gradio demo."""
    print("Starting ASR Server Demo...")
    print("Make sure the ASR server is running: python src/server.py")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
