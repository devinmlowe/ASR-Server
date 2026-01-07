"""ASR Server - FastAPI HTTP server for speech recognition."""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import asyncio
import io
import wave
import struct
import yaml
import uvicorn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, Config
from transcription import Transcriber, TranscriptionResult


# Global instances
transcriber: Transcriber = None
config: Config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global transcriber, config

    # Load configuration
    config = load_config()

    # Resolve model path relative to project root
    project_root = Path(__file__).parent.parent
    model_path = project_root / config.whisper.model_path

    # Initialize transcriber
    transcriber = Transcriber(
        model_path=str(model_path),
        whisper_config=config.whisper,
        vad_config=config.vad
    )

    print(f"ASR Server starting...")
    print(f"  Model: {model_path}")
    print(f"  Language: {config.whisper.language}")
    print(f"  VAD enabled: {config.vad.enabled}")
    print(f"  Listening on: {config.server.host}:{config.server.port}")

    yield

    print("ASR Server shutting down...")


app = FastAPI(
    title="ASR Server",
    description="Local speech recognition server using whisper.cpp",
    version="0.1.0",
    lifespan=lifespan
)


class ConfigUpdate(BaseModel):
    """Request model for config updates."""
    whisper_language: Optional[str] = None
    whisper_threads: Optional[int] = None
    vad_enabled: Optional[bool] = None
    vad_threshold: Optional[float] = None
    vad_min_speech_duration_ms: Optional[int] = None
    vad_min_silence_duration_ms: Optional[int] = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": transcriber is not None,
        "vad_enabled": config.vad.enabled if config else False
    }


@app.get("/v1/config")
async def get_config():
    """Get current server configuration."""
    if config is None:
        raise HTTPException(status_code=503, detail="Config not loaded")

    return {
        "server": {
            "host": config.server.host,
            "port": config.server.port
        },
        "whisper": {
            "model_path": config.whisper.model_path,
            "language": config.whisper.language,
            "threads": config.whisper.threads
        },
        "vad": {
            "enabled": config.vad.enabled,
            "threshold": config.vad.threshold,
            "min_speech_duration_ms": config.vad.min_speech_duration_ms,
            "min_silence_duration_ms": config.vad.min_silence_duration_ms
        }
    }


@app.post("/v1/config")
async def update_config(updates: ConfigUpdate):
    """
    Update server configuration and save to config.yaml.

    Note: Some changes (model_path, host, port) require server restart.
    """
    global config

    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.yaml"

    # Load current config file
    if config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f) or {}
    else:
        config_data = {}

    # Ensure sections exist
    config_data.setdefault("server", {})
    config_data.setdefault("whisper", {})
    config_data.setdefault("vad", {})

    # Apply updates
    changes = []

    if updates.whisper_language is not None:
        config_data["whisper"]["language"] = updates.whisper_language
        config.whisper.language = updates.whisper_language
        changes.append(f"language={updates.whisper_language}")

    if updates.whisper_threads is not None:
        config_data["whisper"]["threads"] = updates.whisper_threads
        config.whisper.threads = updates.whisper_threads
        changes.append(f"threads={updates.whisper_threads}")

    if updates.vad_enabled is not None:
        config_data["vad"]["enabled"] = updates.vad_enabled
        config.vad.enabled = updates.vad_enabled
        changes.append(f"vad_enabled={updates.vad_enabled}")

    if updates.vad_threshold is not None:
        config_data["vad"]["threshold"] = updates.vad_threshold
        config.vad.threshold = updates.vad_threshold
        changes.append(f"vad_threshold={updates.vad_threshold}")

    if updates.vad_min_speech_duration_ms is not None:
        config_data["vad"]["min_speech_duration_ms"] = updates.vad_min_speech_duration_ms
        config.vad.min_speech_duration_ms = updates.vad_min_speech_duration_ms
        changes.append(f"min_speech_duration_ms={updates.vad_min_speech_duration_ms}")

    if updates.vad_min_silence_duration_ms is not None:
        config_data["vad"]["min_silence_duration_ms"] = updates.vad_min_silence_duration_ms
        config.vad.min_silence_duration_ms = updates.vad_min_silence_duration_ms
        changes.append(f"min_silence_duration_ms={updates.vad_min_silence_duration_ms}")

    # Save to file
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)

    return {
        "status": "updated",
        "changes": changes,
        "message": "Config updated. Changes applied to running server and saved to config.yaml."
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(default=None)
):
    """
    Transcribe audio file to text.

    Compatible with OpenAI Whisper API format.

    Args:
        file: Audio file (WAV, MP3, FLAC, OGG supported)
        language: Language code (default: en)

    Returns:
        JSON with transcribed text
    """
    if transcriber is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read audio file
    audio_data = await file.read()

    if len(audio_data) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        result = transcriber.transcribe(audio_data, language)
        return {"text": result.text}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.websocket("/v1/audio/stream")
async def stream_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio transcription.

    Client sends: Binary audio chunks (16kHz, mono, PCM16 or WAV)
    Server sends: JSON with transcription results
    """
    await websocket.accept()

    if transcriber is None:
        await websocket.send_json({"type": "error", "text": "Model not loaded"})
        await websocket.close()
        return

    audio_buffer = bytearray()
    sample_rate = 16000
    silence_threshold_ms = 500  # VAD silence threshold
    min_audio_ms = 100  # Minimum audio to process

    try:
        while True:
            # Receive audio chunk (binary)
            data = await websocket.receive()

            if "bytes" in data:
                audio_chunk = data["bytes"]
                audio_buffer.extend(audio_chunk)

                # Calculate buffer duration
                # Assuming 16-bit mono PCM at 16kHz: 2 bytes per sample
                buffer_duration_ms = (len(audio_buffer) / 2) / sample_rate * 1000

                # Process when we have enough audio (at least 500ms for VAD to work)
                if buffer_duration_ms >= 500:
                    # Convert buffer to WAV format for transcriber
                    wav_data = pcm_to_wav(bytes(audio_buffer), sample_rate)

                    # Run transcription in thread pool to not block
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        transcriber.transcribe,
                        wav_data,
                        config.whisper.language
                    )

                    if result.text.strip():
                        await websocket.send_json({
                            "type": "partial",
                            "text": result.text,
                            "is_utterance_complete": False
                        })

                    # Keep last 100ms for continuity, clear the rest
                    keep_bytes = int(sample_rate * 0.1 * 2)  # 100ms at 16kHz, 16-bit
                    audio_buffer = bytearray(audio_buffer[-keep_bytes:])

            elif "text" in data:
                # Handle control messages
                message = data["text"]
                if message == "end":
                    # Process remaining audio
                    if len(audio_buffer) > int(sample_rate * min_audio_ms / 1000 * 2):
                        wav_data = pcm_to_wav(bytes(audio_buffer), sample_rate)
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None,
                            transcriber.transcribe,
                            wav_data,
                            config.whisper.language
                        )

                        if result.text.strip():
                            await websocket.send_json({
                                "type": "final",
                                "text": result.text,
                                "is_utterance_complete": True
                            })
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "text": str(e)})
    finally:
        await websocket.close()


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """Convert raw PCM16 data to WAV format."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)
    return buffer.getvalue()


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "ASR Server",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "transcribe": "/v1/audio/transcriptions",
            "stream": "/v1/audio/stream (WebSocket)"
        }
    }


def main():
    """Run the server."""
    config = load_config()
    uvicorn.run(
        "server:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
