"""Transcription module using whisper-cli."""

import subprocess
import tempfile
import json
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from config import WhisperConfig, VADConfig

# Supported formats by whisper-cli (no conversion needed)
WHISPER_NATIVE_FORMATS = {'.wav', '.mp3', '.ogg', '.flac'}

# Magic bytes for audio format detection
AUDIO_SIGNATURES = {
    b'RIFF': '.wav',
    b'ID3': '.mp3',
    b'\xff\xfb': '.mp3',
    b'\xff\xfa': '.mp3',
    b'OggS': '.ogg',
    b'fLaC': '.flac',
    b'\x00\x00\x00': '.m4a',  # ftyp box (will check further)
    b'ftyp': '.m4a',
}


def detect_audio_format(data: bytes) -> str:
    """Detect audio format from magic bytes."""
    if data[:4] == b'RIFF':
        return '.wav'
    if data[:3] == b'ID3' or data[:2] in (b'\xff\xfb', b'\xff\xfa'):
        return '.mp3'
    if data[:4] == b'OggS':
        return '.ogg'
    if data[:4] == b'fLaC':
        return '.flac'
    # M4A/MP4/AAC detection (ftyp box)
    if len(data) > 8 and data[4:8] == b'ftyp':
        return '.m4a'
    # CAF (Core Audio Format - iOS native)
    if data[:4] == b'caff':
        return '.caf'
    # Default to m4a for unknown (common from iOS)
    return '.m4a'


def convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convert audio to WAV format using ffmpeg."""
    try:
        result = subprocess.run([
            'ffmpeg', '-y',  # Overwrite output
            '-i', input_path,
            '-ar', '16000',  # 16kHz sample rate (optimal for Whisper)
            '-ac', '1',      # Mono
            '-c:a', 'pcm_s16le',  # 16-bit PCM
            output_path
        ], capture_output=True, timeout=30)
        return result.returncode == 0
    except Exception:
        return False


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
            cmd.append(audio_path)

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
            # Clean up temp files
            Path(temp_path).unlink(missing_ok=True)
            if wav_path:
                Path(wav_path).unlink(missing_ok=True)

    def transcribe_file(self, file_path: str, language: str = None) -> TranscriptionResult:
        """Transcribe audio from a file path."""
        with open(file_path, "rb") as f:
            audio_data = f.read()
        return self.transcribe(audio_data, language)
