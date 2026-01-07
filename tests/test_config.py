# tests/test_config.py
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
