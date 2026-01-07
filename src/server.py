"""ASR Server - FastAPI HTTP server for speech recognition."""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
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


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": transcriber is not None,
        "vad_enabled": config.vad.enabled if config else False
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


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "ASR Server",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "transcribe": "/v1/audio/transcriptions"
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
