# ASR-Server

Local speech recognition server using whisper.cpp + Silero VAD for real-time transcription.

## Features

- CoreML/Metal acceleration on Apple Silicon
- Voice Activity Detection (VAD) for automatic speech boundary detection
- HTTP + SSE API compatible with Whisper API format
- Designed for conversational AI voice mode pipeline

## Architecture

Part of the local AI voice mode stack:
- **ASR-Server** (this) - Speech to text
- **Ollama** - LLM reasoning
- **Chatterbox** - Text to speech

## Quick Start

```bash
# Start the server
cd ~/Documents/git/ASR-Server
source venv/bin/activate
python src/server.py
```

## API

- `POST /v1/audio/transcriptions` - Batch transcription
- `GET /health` - Health check

## Configuration

See `config.yaml` for server settings.

## License

MIT

