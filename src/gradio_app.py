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


def _convert_audio_chunk(audio_tuple) -> bytes:
    """Convert Gradio audio tuple to 16kHz mono PCM bytes."""
    if audio_tuple is None:
        return b""

    sample_rate, audio_data = audio_tuple

    # Convert to mono if stereo
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

    # Normalize and convert to 16-bit PCM
    audio_data = audio_data.astype(np.float32)
    max_val = np.abs(audio_data).max()
    if max_val > 0:
        audio_data = audio_data / max_val
    audio_data = (audio_data * 32767).astype(np.int16)

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


def handle_stream_chunk(audio_chunk, state):
    """
    Handle incoming audio chunk during live recording.
    Called by Gradio's .stream() event every 1 second.

    Returns: (streaming_text, accumulated_audio, state)
    """
    if audio_chunk is None:
        return "Listening...", None, state

    if not check_server_health():
        return "Error: Server not available", None, state

    # Initialize state if needed
    if state is None or not isinstance(state, dict):
        state = {"connected": False, "audio_buffer": bytearray()}

    # Convert chunk to PCM
    pcm_chunk = _convert_audio_chunk(audio_chunk)

    # Accumulate audio in state for batch processing later
    if "audio_buffer" not in state:
        state["audio_buffer"] = bytearray()
    state["audio_buffer"].extend(pcm_chunk)

    # Store the full audio tuple for batch processing
    state["full_audio"] = audio_chunk

    async def process_chunk():
        global _streaming_session

        # Connect on first chunk
        if _streaming_session is None or not _streaming_session.connected:
            await _connect_streaming_session()
            state["connected"] = True

        # Add to session buffer and send
        _streaming_session.audio_buffer.extend(pcm_chunk)
        text = await _send_chunk_to_ws(pcm_chunk)
        return text

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        text = loop.run_until_complete(process_chunk())
        loop.close()

        display_text = text if text else "Listening..."
        return display_text, audio_chunk, state

    except Exception as e:
        return f"Streaming error: {str(e)}", audio_chunk, state


def handle_recording_stop(audio, state, language_override):
    """
    Handle recording stop - finalize streaming and run batch transcription.

    Returns: (streaming_text, streaming_latency, batch_text, batch_latency, state)
    """
    if audio is None:
        return "No audio recorded.", "Latency: --", "No audio recorded.", "Latency: --", {}

    if not check_server_health():
        err = "Error: Server not available"
        return err, "Latency: --", err, "Latency: --", {}

    # Finalize streaming transcription
    async def finalize():
        return await _finalize_streaming()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        streaming_text, streaming_latency = loop.run_until_complete(finalize())
        loop.close()
    except Exception as e:
        streaming_text = f"Streaming error: {str(e)}"
        streaming_latency = 0.0

    # Run batch transcription
    batch_start = time.time()
    batch_text = transcribe_recording_batch(audio, language_override)
    batch_latency = time.time() - batch_start

    # Format latency strings
    streaming_lat_str = f"Latency: {streaming_latency:.2f}s" if streaming_latency > 0 else "Latency: --"
    batch_lat_str = f"Latency: {batch_latency:.2f}s"

    # Reset state for next recording
    global _streaming_session
    _streaming_session = None

    return streaming_text, streaming_lat_str, batch_text, batch_lat_str, {}


def reset_live_outputs():
    """Reset outputs when starting a new recording."""
    global _streaming_session
    _streaming_session = None
    return "Listening...", "Latency: --", "Recording...", "Latency: --", {}


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
                **Real-time comparison:** Streaming transcribes as you speak,
                Batch transcribes after recording stops. Compare accuracy and latency!
                """
            )

            # Audio input with streaming enabled
            recording_input = gr.Audio(
                label="Record Audio (streaming enabled)",
                type="numpy",
                sources=["microphone"],
                streaming=True
            )

            with gr.Row():
                rec_lang_override = gr.Dropdown(
                    choices=[("Use server default", "default")] + LANGUAGES,
                    value="default",
                    label="Language",
                    scale=1
                )

            # Side-by-side outputs
            with gr.Row():
                with gr.Column():
                    streaming_output = gr.Textbox(
                        label="Streaming Output",
                        lines=5,
                        placeholder="Transcription appears in real-time as you speak...",
                        interactive=False
                    )
                    streaming_latency = gr.Markdown("Latency: --")

                with gr.Column():
                    batch_output = gr.Textbox(
                        label="Batch Output",
                        lines=5,
                        placeholder="Transcription appears after recording stops...",
                        interactive=False
                    )
                    batch_latency = gr.Markdown("Latency: --")

            # Hidden state for accumulated audio
            live_state = gr.State(value={})
            # Hidden component to accumulate audio for batch processing
            accumulated_audio = gr.State(value=None)

            # Stream event - fires every 1 second during recording
            recording_input.stream(
                fn=handle_stream_chunk,
                inputs=[recording_input, live_state],
                outputs=[streaming_output, accumulated_audio, live_state],
                stream_every=1,
                time_limit=300  # 5 minute max recording
            )

            # Stop recording event - finalize streaming and run batch
            recording_input.stop_recording(
                fn=handle_recording_stop,
                inputs=[accumulated_audio, live_state, rec_lang_override],
                outputs=[streaming_output, streaming_latency, batch_output, batch_latency, live_state]
            )

            # Start recording event - reset outputs
            recording_input.start_recording(
                fn=reset_live_outputs,
                outputs=[streaming_output, streaming_latency, batch_output, batch_latency, live_state]
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
