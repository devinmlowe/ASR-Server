# Live Voice Mode Architecture

**Date:** 2026-01-08
**Status:** Design / Planning

## Overview

The VAD Live Transcription tab (`src/gradio_app.py`) serves as a foundational component for a larger **Live Voice Mode** system. This document captures the intended architecture and integration points.

## Current Implementation

The VAD Live Transcription provides:
- Client-side RMS energy detection for speech/silence classification
- Auto-transcription when pause detected after speech (configurable threshold)
- Rolling transcript with timestamps
- Statistics display (Speaking, Silence duration, RMS by period)

**Key Parameters (tuned for current environment):**
- Silence Threshold (RMS): 0.25
- Min Silence Duration: 700ms

## Target Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Microphone    │────▶│  VAD Detection  │────▶│  Transcription  │
│   (Gradio)      │     │  (RMS-based)    │     │  (Whisper)      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   LLM Response  │◀────│  Message Queue  │
                        │   (Ollama)      │     │                 │
                        └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   TTS Output    │
                        │   (optional)    │
                        └─────────────────┘
```

## Pipeline Flow

1. **VAD Detection**: Client-side RMS monitoring detects speech boundaries
2. **Pause Detection**: When silence exceeds threshold after speech, utterance is complete
3. **Transcription**: Audio chunk sent to Whisper API, text returned
4. **Message Queue**: Transcribed text pushed to queue for LLM processing
5. **LLM Processing**: Ollama (or other LLM) receives text, generates response
6. **Response Handling**: LLM output displayed and/or spoken via TTS

## Integration Points

### Message Queue Options
- Redis pub/sub
- Python `queue.Queue` (in-process)
- ZeroMQ
- Simple HTTP webhook

### Ollama Integration
- Local Ollama server (default: `http://localhost:11434`)
- `/api/generate` or `/api/chat` endpoints
- Streaming responses for low latency

## Next Steps

1. Add message queue output to VAD handler (after transcription)
2. Create Ollama client module
3. Wire up response display in UI
4. Optional: Add TTS for spoken responses

## Notes

- The VAD implementation clears audio buffer before transcription to prevent word loss at utterance boundaries
- Statistics tracking helps tune threshold for different environments/speakers
