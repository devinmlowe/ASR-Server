"""Configuration loader for ASR Server."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8005


@dataclass
class WhisperConfig:
    model_path: str = "models/ggml-base.bin"
    language: str = "en"
    threads: int = 4


@dataclass
class VADConfig:
    enabled: bool = True
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 700


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    vad: VADConfig = field(default_factory=VADConfig)


def load_config(path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    config_path = Path(path)

    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        return Config(
            server=ServerConfig(**data.get("server", {})),
            whisper=WhisperConfig(**data.get("whisper", {})),
            vad=VADConfig(**data.get("vad", {}))
        )

    return Config()
