# Gradio Frontend Design

## Overview

Add a minimal Gradio-based testing/demo interface for the ASR Server with file upload and real-time recording capabilities.

## Architecture

```
┌─────────────────────────────────────────┐
│  gradio_app.py (standalone)             │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  File Upload Tab                │    │
│  │  - Drop audio file              │    │
│  │  - Transcribe button            │    │
│  │  - Text output                  │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  Live Recording Tab             │    │
│  │  - Mode toggle: Batch/Stream    │    │
│  │  - Record button                │    │
│  │  - Text output                  │    │
│  └─────────────────────────────────┘    │
│                                         │
│  [API Docs link → /docs]                │
└────────────────┬────────────────────────┘
                 │ HTTP / WebSocket
                 ▼
┌─────────────────────────────────────────┐
│  server.py (existing FastAPI)           │
│  + New: /v1/audio/stream (WebSocket)    │
└─────────────────────────────────────────┘
```

## File Changes

| File | Change |
|------|--------|
| `src/gradio_app.py` | New - standalone Gradio UI |
| `src/server.py` | Add WebSocket `/v1/audio/stream` endpoint |
| `requirements.txt` | Add `gradio` |

## UI Layout

```
┌─────────────────────────────────────────────────────┐
│  ASR Server Demo                    [API Docs ↗]   │
├─────────────────────────────────────────────────────┤
│  [File Upload]  [Live Recording]                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Audio Input                                 │   │
│  │  (file dropzone or mic recorder)            │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  [ Transcribe ]                                     │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Transcription Output                        │   │
│  │  (text appears here)                         │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Status: Ready                                      │
└─────────────────────────────────────────────────────┘
```

## WebSocket Streaming Endpoint

**Endpoint:** `WebSocket /v1/audio/stream`

**Client sends:** Binary audio chunks (16kHz, mono, PCM or WAV)

**Server sends:** JSON messages
```json
{
  "type": "partial" | "final",
  "text": "transcribed text",
  "is_utterance_complete": true | false
}
```

**Flow:**
1. Client opens WebSocket connection
2. Client streams audio chunks (e.g., every 500ms)
3. Server buffers audio, runs VAD + Whisper on speech segments
4. Server sends back partial/final transcriptions
5. `is_utterance_complete: true` signals VAD detected end of speech
6. Client closes connection when done

## Live Recording Modes

- **Batch mode:** Record → Stop → Transcribe (uses existing HTTP POST endpoint)
- **Streaming mode:** Transcribes continuously while recording (uses WebSocket)

## Decisions

- **Standalone Gradio app** - Keeps production API clean, clear separation
- **Server-side VAD only** - No client-side VAD; server VAD already protects GPU
- **Minimal output format** - Keep HTTP endpoint OpenAI-compatible (`{"text": "..."}`)
- **WebSocket for streaming** - Standard approach, good Gradio support

## Out of Scope (YAGNI)

- Client-side VAD
- Enriched HTTP response fields
- Complex UI styling
- Multiple language selection UI (uses config default)
