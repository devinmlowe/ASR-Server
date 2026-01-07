"""Transcription module using whisper-cli."""

import subprocess
import tempfile
import json
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from config import WhisperConfig, VADConfig


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration_ms: float
    segments: list = None


class Transcriber:
    """Wrapper around whisper-cli for speech-to-text transcription."""

    def __init__(
        self,
        model_path: str,
        whisper_config: WhisperConfig,
        vad_config: VADConfig
    ):
        self.model_path = Path(model_path).resolve()
        self.config = whisper_config
        self.vad_config = vad_config

        # Verify model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Whisper model not found: {self.model_path}")

    def transcribe(self, audio_data: bytes, language: str = None) -> TranscriptionResult:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes (WAV format)
            language: Language code (default: from config)

        Returns:
            TranscriptionResult with transcribed text
        """
        language = language or self.config.language

        # Write audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            # Build whisper-cli command
            cmd = [
                "whisper-cli",
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
            cmd.append(temp_path)

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
                duration_ms=0,  # Could parse from whisper timing output if needed
                segments=None
            )

        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    def transcribe_file(self, file_path: str, language: str = None) -> TranscriptionResult:
        """Transcribe audio from a file path."""
        with open(file_path, "rb") as f:
            audio_data = f.read()
        return self.transcribe(audio_data, language)
