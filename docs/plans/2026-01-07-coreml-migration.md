# CoreML Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate ASR-Server from Homebrew whisper-cli (Metal) to a custom-built whisper.cpp with CoreML support for Neural Engine acceleration, while maintaining the ability to switch back to Metal.

**Architecture:** Build whisper.cpp from source with CoreML enabled, generate CoreML encoder models alongside existing GGML models, and add a configurable backend selector. The transcription module will detect which backend to use based on config and model availability.

**Tech Stack:** whisper.cpp (C++), CoreML, Python 3, FastAPI, CMake, Xcode Command Line Tools

---

## Prerequisites

Before starting, ensure you have:
- Xcode Command Line Tools installed (`xcode-select --install`)
- CMake installed (`brew install cmake`)
- Python 3.10+ with pip

---

## Task 1: Clone and Build whisper.cpp with CoreML

**Files:**
- Create: `vendor/whisper.cpp/` (git clone)
- Create: `vendor/whisper.cpp/build/` (cmake build)
- Create: `bin/whisper-cli` (symlink to built binary)

**Step 1: Clone whisper.cpp repository**

```bash
cd /Users/devinmlowe/Documents/git/ASR-Server
mkdir -p vendor
git clone https://github.com/ggerganov/whisper.cpp.git vendor/whisper.cpp
```

Expected: Repository cloned to `vendor/whisper.cpp/`

**Step 2: Build whisper.cpp with CoreML and Metal support**

```bash
cd /Users/devinmlowe/Documents/git/ASR-Server/vendor/whisper.cpp
cmake -B build \
  -DWHISPER_COREML=1 \
  -DWHISPER_METAL=1 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

Expected: Build completes with `whisper-cli` binary in `build/bin/`

**Step 3: Verify the built binary works**

```bash
./vendor/whisper.cpp/build/bin/whisper-cli --help
```

Expected: Help output showing available options including `--coreml`

**Step 4: Create symlink for project access**

```bash
mkdir -p /Users/devinmlowe/Documents/git/ASR-Server/bin
ln -sf ../vendor/whisper.cpp/build/bin/whisper-cli /Users/devinmlowe/Documents/git/ASR-Server/bin/whisper-cli
```

Expected: `bin/whisper-cli` symlink created

**Step 5: Add vendor/ and bin/ to .gitignore**

```bash
echo -e "\n# Local builds\nvendor/\nbin/" >> /Users/devinmlowe/Documents/git/ASR-Server/.gitignore
```

Expected: `.gitignore` updated

**Step 6: Commit**

```bash
git add .gitignore
git commit -m "build: add vendor/ and bin/ to gitignore for local whisper.cpp build"
```

---

## Task 2: Generate CoreML Model

**Files:**
- Create: `models/ggml-base-encoder.mlmodelc/` (CoreML model directory)

**Step 1: Install Python dependencies for CoreML model generation**

```bash
cd /Users/devinmlowe/Documents/git/ASR-Server
source venv/bin/activate
pip install ane_transformers openai-whisper coremltools
```

Expected: Packages installed successfully

**Step 2: Generate CoreML encoder model for base**

```bash
cd /Users/devinmlowe/Documents/git/ASR-Server/vendor/whisper.cpp
bash ./models/generate-coreml-model.sh base
```

Expected: Creates `models/ggml-base-encoder.mlmodelc/` directory

**Step 3: Move CoreML model to project models directory**

```bash
mv /Users/devinmlowe/Documents/git/ASR-Server/vendor/whisper.cpp/models/ggml-base-encoder.mlmodelc \
   /Users/devinmlowe/Documents/git/ASR-Server/models/
```

Expected: `models/ggml-base-encoder.mlmodelc/` exists alongside `models/ggml-base.bin`

**Step 4: Verify CoreML model structure**

```bash
ls -la /Users/devinmlowe/Documents/git/ASR-Server/models/ggml-base-encoder.mlmodelc/
```

Expected: Directory contains compiled CoreML model files

**Step 5: Test CoreML inference**

```bash
cd /Users/devinmlowe/Documents/git/ASR-Server
# Download a test audio file if none exists
curl -sL "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav" -o /tmp/test.wav
./bin/whisper-cli -m models/ggml-base.bin /tmp/test.wav --no-timestamps -np
```

Expected: Transcription output (CoreML will auto-detect if `.mlmodelc` is present alongside `.bin`)

---

## Task 3: Update Configuration Schema

**Files:**
- Modify: `src/config.py:16-19` (WhisperConfig dataclass)
- Modify: `config.yaml:5-8` (whisper section)

**Step 1: Write the test for new config fields**

Create test file:

```python
# tests/test_config.py
import pytest
from config import WhisperConfig, load_config

def test_whisper_config_has_backend_field():
    """WhisperConfig should have backend field with default 'auto'."""
    config = WhisperConfig()
    assert hasattr(config, 'backend')
    assert config.backend == 'auto'

def test_whisper_config_has_cli_path_field():
    """WhisperConfig should have cli_path field with default None."""
    config = WhisperConfig()
    assert hasattr(config, 'cli_path')
    assert config.cli_path is None

def test_whisper_config_backend_values():
    """Backend should accept 'auto', 'coreml', or 'metal'."""
    for backend in ['auto', 'coreml', 'metal']:
        config = WhisperConfig(backend=backend)
        assert config.backend == backend
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/devinmlowe/Documents/git/ASR-Server
source venv/bin/activate
python -m pytest tests/test_config.py -v
```

Expected: FAIL with `AttributeError: 'WhisperConfig' object has no attribute 'backend'`

**Step 3: Update WhisperConfig dataclass**

Modify `src/config.py` - replace the WhisperConfig class:

```python
@dataclass
class WhisperConfig:
    model_path: str = "models/ggml-base.bin"
    language: str = "en"
    threads: int = 4
    backend: str = "auto"  # "auto", "coreml", or "metal"
    cli_path: Optional[str] = None  # Path to whisper-cli binary, None = search PATH
```

Also add `Optional` import at top:

```python
from typing import Optional
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_config.py -v
```

Expected: PASS

**Step 5: Update config.yaml with new fields**

Modify `config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8005

whisper:
  model_path: "models/ggml-base.bin"
  language: "en"
  threads: 4
  backend: "auto"  # "auto", "coreml", or "metal"
  cli_path: "bin/whisper-cli"  # Use local build, or null for system whisper-cli

vad:
  enabled: false
  threshold: 0.5
  min_speech_duration_ms: 250
  min_silence_duration_ms: 700
```

**Step 6: Commit**

```bash
git add src/config.py config.yaml tests/test_config.py
git commit -m "feat: add backend and cli_path config options for CoreML/Metal switching"
```

---

## Task 4: Update Transcriber to Support Backend Selection

**Files:**
- Modify: `src/transcription.py:74-90` (Transcriber.__init__)
- Modify: `src/transcription.py:124-131` (whisper-cli command building)

**Step 1: Write the test for backend selection**

Add to tests:

```python
# tests/test_transcription.py
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from config import WhisperConfig, VADConfig
from transcription import Transcriber

@pytest.fixture
def mock_model_file():
    """Create a temporary mock model file."""
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        f.write(b'mock model data')
        yield f.name
    os.unlink(f.name)

def test_transcriber_resolves_cli_path(mock_model_file):
    """Transcriber should resolve cli_path from config."""
    whisper_config = WhisperConfig(cli_path="/usr/local/bin/whisper-cli")
    vad_config = VADConfig()

    transcriber = Transcriber(
        model_path=mock_model_file,
        whisper_config=whisper_config,
        vad_config=vad_config
    )

    assert transcriber.cli_path == "/usr/local/bin/whisper-cli"

def test_transcriber_default_cli_path(mock_model_file):
    """Transcriber should default to 'whisper-cli' when cli_path is None."""
    whisper_config = WhisperConfig(cli_path=None)
    vad_config = VADConfig()

    transcriber = Transcriber(
        model_path=mock_model_file,
        whisper_config=whisper_config,
        vad_config=vad_config
    )

    assert transcriber.cli_path == "whisper-cli"

def test_transcriber_stores_backend(mock_model_file):
    """Transcriber should store backend from config."""
    whisper_config = WhisperConfig(backend="coreml")
    vad_config = VADConfig()

    transcriber = Transcriber(
        model_path=mock_model_file,
        whisper_config=whisper_config,
        vad_config=vad_config
    )

    assert transcriber.backend == "coreml"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_transcription.py -v
```

Expected: FAIL with `AttributeError: 'Transcriber' object has no attribute 'cli_path'`

**Step 3: Update Transcriber.__init__ method**

Modify `src/transcription.py` - update the `__init__` method:

```python
def __init__(
    self,
    model_path: str,
    whisper_config: WhisperConfig,
    vad_config: VADConfig
):
    self.model_path = Path(model_path).resolve()
    self.config = whisper_config
    self.vad_config = vad_config

    # Backend configuration
    self.backend = whisper_config.backend
    self.cli_path = whisper_config.cli_path or "whisper-cli"

    # Verify model exists
    if not self.model_path.exists():
        raise FileNotFoundError(f"Whisper model not found: {self.model_path}")

    # Check for CoreML model if backend is auto or coreml
    self.coreml_model_path = self._find_coreml_model()

def _find_coreml_model(self) -> Optional[Path]:
    """Find the CoreML encoder model alongside the GGML model."""
    # CoreML model naming: ggml-base.bin -> ggml-base-encoder.mlmodelc
    model_name = self.model_path.stem  # e.g., "ggml-base"
    coreml_path = self.model_path.parent / f"{model_name}-encoder.mlmodelc"

    if coreml_path.exists() and coreml_path.is_dir():
        return coreml_path
    return None
```

Also add `Optional` to the imports at top if not already there:

```python
from typing import Optional, Tuple
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_transcription.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/transcription.py tests/test_transcription.py
git commit -m "feat: add backend and cli_path support to Transcriber"
```

---

## Task 5: Update Command Building for Backend Selection

**Files:**
- Modify: `src/transcription.py:124-144` (transcribe method command building)

**Step 1: Write the test for command building**

Add to `tests/test_transcription.py`:

```python
def test_transcriber_builds_coreml_command(mock_model_file, tmp_path):
    """Transcriber should not add extra flags for CoreML (auto-detected by model presence)."""
    whisper_config = WhisperConfig(backend="coreml", cli_path="whisper-cli")
    vad_config = VADConfig(enabled=False)

    transcriber = Transcriber(
        model_path=mock_model_file,
        whisper_config=whisper_config,
        vad_config=vad_config
    )

    # Build command (we need to expose this for testing)
    cmd = transcriber._build_command("/tmp/test.wav", "en")

    assert cmd[0] == "whisper-cli"
    assert "-m" in cmd
    assert "--no-timestamps" in cmd
    assert "-np" in cmd

def test_transcriber_uses_custom_cli_path(mock_model_file):
    """Transcriber should use custom cli_path in command."""
    whisper_config = WhisperConfig(cli_path="/custom/path/whisper-cli")
    vad_config = VADConfig(enabled=False)

    transcriber = Transcriber(
        model_path=mock_model_file,
        whisper_config=whisper_config,
        vad_config=vad_config
    )

    cmd = transcriber._build_command("/tmp/test.wav", "en")

    assert cmd[0] == "/custom/path/whisper-cli"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_transcription.py::test_transcriber_builds_coreml_command -v
```

Expected: FAIL with `AttributeError: 'Transcriber' object has no attribute '_build_command'`

**Step 3: Refactor transcribe method to extract command building**

Modify `src/transcription.py` - add `_build_command` method and update `transcribe`:

```python
def _build_command(self, audio_path: str, language: str) -> list:
    """Build the whisper-cli command."""
    cmd = [
        self.cli_path,
        "-m", str(self.model_path),
        "-l", language,
        "-t", str(self.config.threads),
        "--no-timestamps",
        "-np",  # No prints (only output)
    ]

    # Add VAD options if enabled
    if self.vad_config.enabled:
        cmd.extend([
            "--vad",
            "-vt", str(self.vad_config.threshold),
            "-vspd", str(self.vad_config.min_speech_duration_ms),
            "-vsd", str(self.vad_config.min_silence_duration_ms),
        ])

    # Add input file
    cmd.append(audio_path)

    return cmd

def transcribe(self, audio_data: bytes, language: str = None) -> TranscriptionResult:
    """
    Transcribe audio data to text.

    Args:
        audio_data: Raw audio bytes (any supported format)
        language: Language code (default: from config)

    Returns:
        TranscriptionResult with transcribed text
    """
    language = language or self.config.language

    # Detect audio format from magic bytes
    detected_format = detect_audio_format(audio_data)
    needs_conversion = detected_format not in WHISPER_NATIVE_FORMATS

    # Write audio to temp file with correct extension
    with tempfile.NamedTemporaryFile(suffix=detected_format, delete=False) as f:
        f.write(audio_data)
        temp_path = f.name

    wav_path = None
    try:
        # Convert to WAV if needed
        if needs_conversion:
            wav_path = temp_path.replace(detected_format, '.wav')
            if not convert_to_wav(temp_path, wav_path):
                raise RuntimeError(f"Failed to convert {detected_format} to WAV")
            audio_path = wav_path
        else:
            audio_path = temp_path

        # Build command using extracted method
        cmd = self._build_command(audio_path, language)

        # Run whisper-cli
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"whisper-cli failed: {result.stderr}")

        # Parse output - whisper-cli outputs text directly
        text = result.stdout.strip()

        # Clean up any remaining timestamp artifacts
        text = re.sub(r'\[[\d:\.]+\s*-->\s*[\d:\.]+\]\s*', '', text)
        text = text.strip()

        return TranscriptionResult(
            text=text,
            language=language,
            duration_ms=0,
            segments=None
        )

    finally:
        # Clean up temp files
        Path(temp_path).unlink(missing_ok=True)
        if wav_path:
            Path(wav_path).unlink(missing_ok=True)
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_transcription.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/transcription.py tests/test_transcription.py
git commit -m "refactor: extract _build_command method for testability"
```

---

## Task 6: Update Server Config Endpoint

**Files:**
- Modify: `src/server.py:88-110` (get_config endpoint)
- Modify: `src/server.py:68-76` (ConfigUpdate model)

**Step 1: Write the test for config endpoint**

Create test file:

```python
# tests/test_server.py
import pytest
from fastapi.testclient import TestClient

def test_config_includes_backend(test_client):
    """GET /v1/config should include backend field."""
    response = test_client.get("/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert "whisper" in data
    assert "backend" in data["whisper"]
    assert data["whisper"]["backend"] in ["auto", "coreml", "metal"]
```

Note: This test assumes a test fixture for the FastAPI client exists. For simplicity, we'll add the functionality and test manually.

**Step 2: Update ConfigUpdate model**

Modify `src/server.py` - update the ConfigUpdate class:

```python
class ConfigUpdate(BaseModel):
    """Request model for config updates."""
    whisper_language: Optional[str] = None
    whisper_threads: Optional[int] = None
    whisper_backend: Optional[str] = None  # New field
    vad_enabled: Optional[bool] = None
    vad_threshold: Optional[float] = None
    vad_min_speech_duration_ms: Optional[int] = None
    vad_min_silence_duration_ms: Optional[int] = None
```

**Step 3: Update get_config endpoint**

Modify `src/server.py` - update the get_config function:

```python
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
            "threads": config.whisper.threads,
            "backend": config.whisper.backend,
            "cli_path": config.whisper.cli_path
        },
        "vad": {
            "enabled": config.vad.enabled,
            "threshold": config.vad.threshold,
            "min_speech_duration_ms": config.vad.min_speech_duration_ms,
            "min_silence_duration_ms": config.vad.min_silence_duration_ms
        }
    }
```

**Step 4: Update update_config endpoint**

Add backend update handling to the `update_config` function (around line 140):

```python
if updates.whisper_backend is not None:
    if updates.whisper_backend not in ["auto", "coreml", "metal"]:
        raise HTTPException(status_code=400, detail="backend must be 'auto', 'coreml', or 'metal'")
    config_data["whisper"]["backend"] = updates.whisper_backend
    config.whisper.backend = updates.whisper_backend
    changes.append(f"backend={updates.whisper_backend}")
```

**Step 5: Test manually**

```bash
cd /Users/devinmlowe/Documents/git/ASR-Server
source venv/bin/activate
python src/server.py &
sleep 3
curl http://localhost:8005/v1/config | jq .
kill %1
```

Expected: Response includes `"backend": "auto"` in whisper section

**Step 6: Commit**

```bash
git add src/server.py
git commit -m "feat: add backend field to config API endpoints"
```

---

## Task 7: Update Gradio UI Settings

**Files:**
- Modify: `src/gradio_app.py:58-91` (get_config_display)
- Modify: `src/gradio_app.py:403-456` (Settings tab)

**Step 1: Update get_config_display function**

Modify `src/gradio_app.py` - update the function to return backend info:

```python
def get_config_display():
    """Get formatted config for display."""
    config = load_server_config()
    if config is None:
        return "Server not available", "en", 4, "auto", False, 0.5, 250, 700

    whisper = config.get("whisper", {})
    vad = config.get("vad", {})

    backend = whisper.get('backend', 'auto')
    backend_display = {
        'auto': 'Auto (CoreML if available, else Metal)',
        'coreml': 'CoreML (Neural Engine)',
        'metal': 'Metal (GPU)'
    }.get(backend, backend)

    status = f"""**Server Status:** Connected

**Whisper Settings:**
- Model: `{whisper.get('model_path', 'unknown')}`
- Language: `{whisper.get('language', 'en')}`
- Threads: `{whisper.get('threads', 4)}`
- Backend: `{backend_display}`
- CLI Path: `{whisper.get('cli_path', 'system default')}`

**VAD Settings:**
- Enabled: `{vad.get('enabled', False)}`
- Threshold: `{vad.get('threshold', 0.5)}`
- Min Speech Duration: `{vad.get('min_speech_duration_ms', 250)}ms`
- Min Silence Duration: `{vad.get('min_silence_duration_ms', 700)}ms`
"""

    return (
        status,
        whisper.get('language', 'en'),
        whisper.get('threads', 4),
        whisper.get('backend', 'auto'),
        vad.get('enabled', False),
        vad.get('threshold', 0.5),
        vad.get('min_speech_duration_ms', 250),
        vad.get('min_silence_duration_ms', 700)
    )
```

**Step 2: Update save_config function signature**

Add backend parameter:

```python
def save_config(language, threads, backend, vad_enabled, vad_threshold, min_speech_ms, min_silence_ms):
    """Save config to server."""
    if not check_server_health():
        return "Error: Server not available"

    try:
        payload = {
            "whisper_language": language,
            "whisper_threads": int(threads),
            "whisper_backend": backend,
            "vad_enabled": vad_enabled,
            "vad_threshold": float(vad_threshold),
            "vad_min_speech_duration_ms": int(min_speech_ms),
            "vad_min_silence_duration_ms": int(min_silence_ms)
        }
        # ... rest unchanged
```

**Step 3: Add backend dropdown to Settings tab**

In the Settings tab section, add after threads slider:

```python
settings_backend = gr.Dropdown(
    choices=[
        ("Auto (CoreML if available)", "auto"),
        ("CoreML (Neural Engine)", "coreml"),
        ("Metal (GPU)", "metal")
    ],
    value="auto",
    label="Inference Backend"
)
```

**Step 4: Update all UI callbacks to include backend**

Update `refresh_btn.click`, `save_config_btn.click`, and `demo.load` outputs/inputs to include `settings_backend`.

**Step 5: Commit**

```bash
git add src/gradio_app.py
git commit -m "feat: add backend selection to Gradio UI settings"
```

---

## Task 8: Integration Testing

**Files:**
- None (testing only)

**Step 1: Start the server with new config**

```bash
cd /Users/devinmlowe/Documents/git/ASR-Server
./asr.sh start
```

Expected: Server starts without errors

**Step 2: Test batch transcription**

```bash
curl -X POST http://localhost:8005/v1/audio/transcriptions \
  -F "file=@/tmp/test.wav" \
  | jq .
```

Expected: JSON response with `"text"` field containing transcription

**Step 3: Test config endpoint**

```bash
curl http://localhost:8005/v1/config | jq '.whisper.backend'
```

Expected: `"auto"`

**Step 4: Test backend switching**

```bash
curl -X POST http://localhost:8005/v1/config \
  -H "Content-Type: application/json" \
  -d '{"whisper_backend": "metal"}' \
  | jq .
```

Expected: `"changes": ["backend=metal"]`

**Step 5: Test Gradio UI**

Open http://localhost:7860 in browser
- Navigate to Settings tab
- Verify Backend dropdown shows "Auto (CoreML if available)"
- Change to "Metal (GPU)" and save
- Verify change persists after refresh

**Step 6: Performance comparison (optional)**

```bash
# Test with Metal
curl -X POST http://localhost:8005/v1/config -H "Content-Type: application/json" -d '{"whisper_backend": "metal"}'
time curl -X POST http://localhost:8005/v1/audio/transcriptions -F "file=@/tmp/test.wav" -o /dev/null

# Test with CoreML
curl -X POST http://localhost:8005/v1/config -H "Content-Type: application/json" -d '{"whisper_backend": "coreml"}'
time curl -X POST http://localhost:8005/v1/audio/transcriptions -F "file=@/tmp/test.wav" -o /dev/null
```

Note: First CoreML run will be slower due to ANE compilation. Subsequent runs should be faster.

---

## Task 9: Documentation and Rollback Strategy

**Files:**
- Modify: `README.md` (if exists) or create notes

**Rollback Strategy:**

To revert to Homebrew whisper-cli:

1. Update `config.yaml`:
   ```yaml
   whisper:
     cli_path: null  # Use system whisper-cli
     backend: "metal"
   ```

2. Restart server:
   ```bash
   ./asr.sh restart
   ```

The system will use the Homebrew-installed `whisper-cli` with Metal backend.

**Step 1: Commit final changes**

```bash
git add -A
git commit -m "docs: add CoreML migration documentation and rollback strategy"
```

---

## Summary

After completing all tasks:

1. **whisper.cpp** built from source in `vendor/whisper.cpp/` with CoreML+Metal
2. **CoreML model** generated at `models/ggml-base-encoder.mlmodelc/`
3. **Config** supports `backend` ("auto"/"coreml"/"metal") and `cli_path` options
4. **Transcriber** uses custom CLI path and respects backend setting
5. **API** exposes backend config via GET/POST `/v1/config`
6. **Gradio UI** has backend dropdown in Settings
7. **Rollback** possible by setting `cli_path: null` and `backend: metal`

Expected performance improvement: ~3x faster with CoreML on Apple Silicon (after first-run ANE compilation).
