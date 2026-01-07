# Live Streaming UI Enhancement Design

**Date:** 2026-01-07
**Status:** Approved
**Goal:** Add side-by-side Batch vs Streaming transcription comparison with true real-time streaming during recording.

---

## Overview

Enhance the Gradio "Live Recording" tab to show both Batch and Streaming transcription results simultaneously, with latency metrics for performance comparison. The Streaming output updates in real-time AS the user records, not after.

---

## UI Layout

```
┌─────────────────────────────────────────────────────────┐
│  🎤 Record Audio                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │        [Microphone Recording Component]          │   │
│  │            (streaming=True)                      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Language: [Dropdown - Use server default ▼]           │
│                                                         │
│  ┌─────────────────────┐   ┌─────────────────────┐     │
│  │ Streaming Output    │   │ Batch Output        │     │
│  │                     │   │                     │     │
│  │ "The transcribed    │   │ "The transcribed    │     │
│  │  text appears here  │   │  text appears here  │     │
│  │  in real-time..."   │   │  after recording."  │     │
│  │                     │   │                     │     │
│  └─────────────────────┘   └─────────────────────┘     │
│  Latency: 0.3s              Latency: 1.2s              │
└─────────────────────────────────────────────────────────┘
```

**Key changes from current UI:**
- Remove the "Batch/Streaming" radio toggle (both run simultaneously)
- Remove the "Transcribe" button (streaming happens automatically; batch triggers on stop)
- Two side-by-side textboxes instead of one
- Latency labels below each textbox

---

## Data Flow & Behavior

### Recording Lifecycle

1. **User clicks record** → Microphone starts capturing
   - Streaming textbox shows "Listening..." placeholder
   - Batch textbox shows "Recording..." placeholder
   - Both latency labels reset to empty

2. **Every 1 second during recording** → Gradio's `.stream()` fires
   - Audio chunk sent to WebSocket `/v1/audio/stream`
   - Server accumulates buffer, re-transcribes full audio every ~2s
   - Streaming textbox updates with latest partial transcription
   - (No latency shown yet - still recording)

3. **User stops recording** → Two parallel actions:
   - **Streaming path**: Send "end" to WebSocket, receive final result, record latency
   - **Batch path**: Send full audio to HTTP `/v1/audio/transcriptions`, record latency

4. **Both complete** → Display final results with latency metrics

### Latency Measurement

- **Metric**: Time from "stop recording" to transcription result received
- **Display**: Simple text below each textbox (e.g., "Latency: 1.2s")
- **Why**: Apples-to-apples comparison between streaming and batch modes

### State Management

- `gr.State()` for accumulated audio buffer (streaming)
- `gr.State()` for recording start/stop timestamps (latency calculation)
- Streaming and Batch operate independently (no shared state beyond audio)

### Error Handling

- If WebSocket disconnects mid-recording: Show error in Streaming box, Batch still processes
- If server unavailable: Both boxes show error message

---

## Implementation Details

### Gradio Components

```python
with gr.TabItem("Live Recording"):
    # Single streaming audio input
    recording_input = gr.Audio(
        sources=["microphone"],
        streaming=True,        # Enable real-time streaming
        type="numpy"
    )

    # Language selector (shared by both)
    rec_lang_override = gr.Dropdown(...)

    # Side-by-side outputs
    with gr.Row():
        with gr.Column():
            streaming_output = gr.Textbox(label="Streaming", lines=5)
            streaming_latency = gr.Markdown("Latency: --")
        with gr.Column():
            batch_output = gr.Textbox(label="Batch", lines=5)
            batch_latency = gr.Markdown("Latency: --")

    # State for audio accumulation and timing
    audio_state = gr.State(value=None)
    timing_state = gr.State(value={})
```

### Event Handlers

1. **`.stream()` event** - fires every 1 second during recording
   - Accumulates audio chunks
   - Sends to WebSocket for real-time transcription
   - Updates streaming textbox

2. **`.stop_recording()` event** - fires when user stops
   - Triggers batch transcription via HTTP API
   - Finalizes streaming transcription
   - Calculates and displays both latencies

### Streaming Interval

- Gradio `stream_every=1` (1 second chunks)
- Server buffers 2 seconds before re-transcribing
- Users see updates roughly every 2 seconds

---

## WebSocket Connection Management

The trickiest part is maintaining a WebSocket connection across multiple `.stream()` calls.

### Solution: Session Manager Class

```python
class StreamingSession:
    def __init__(self):
        self.ws = None
        self.connected = False
        self.latest_text = ""
        self.start_time = None

    async def connect(self):
        self.ws = await websockets.connect(WS_URL)
        self.connected = True
        self.start_time = time.time()

    async def send_chunk(self, pcm_data):
        if self.connected:
            await self.ws.send(pcm_data)
            # Non-blocking check for responses
            try:
                response = await asyncio.wait_for(self.ws.recv(), timeout=0.05)
                msg = json.loads(response)
                if msg.get("text"):
                    self.latest_text = msg["text"]
            except asyncio.TimeoutError:
                pass

    async def finalize(self):
        await self.ws.send("end")
        # Drain until final message received
        while True:
            response = await asyncio.wait_for(self.ws.recv(), timeout=10)
            msg = json.loads(response)
            if msg.get("text"):
                self.latest_text = msg["text"]
            if msg.get("type") == "final":
                break
        latency = time.time() - self.stop_time
        return self.latest_text, latency
```

### Lifecycle

1. First `.stream()` call → Create session, connect WebSocket
2. Subsequent `.stream()` calls → Send chunks, poll for updates
3. `.stop_recording()` → Finalize, get latency, close connection

---

## Files to Modify

- `src/gradio_app.py` - Main UI changes (lines 369-402 primarily)

## Dependencies

- No new dependencies required
- Uses existing `websockets` and `asyncio` imports

---

## Testing Plan

1. **Manual testing**: Record audio, verify both outputs update correctly
2. **Latency verification**: Compare displayed latencies with stopwatch
3. **Error scenarios**: Test WebSocket disconnect, server unavailable
4. **Long recordings**: Test 30+ second recordings for stability
