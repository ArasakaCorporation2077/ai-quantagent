"""Configuration loader."""

import os
from pathlib import Path

import yaml


def _find_project_root() -> Path:
    """Find the project root by looking for config/ directory."""
    current = Path(__file__).resolve().parent.parent
    if (current / 'config').exists():
        return current
    return Path(os.getcwd())


PROJECT_ROOT = _find_project_root()


def load_config(config_path: str | None = None) -> dict:
    """Load the main configuration file."""
    if config_path is None:
        config_path = PROJECT_ROOT / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_secrets(secrets_path: str | None = None) -> dict:
    """Load the secrets file with API keys."""
    if secrets_path is None:
        secrets_path = PROJECT_ROOT / 'config' / 'secrets.yaml'
    if not os.path.exists(secrets_path):
        return {}
    with open(secrets_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def get_data_dir(config: dict) -> Path:
    """Resolve data directory path."""
    data_dir = config['project']['data_dir']
    p = Path(data_dir)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_db_path(config: dict) -> Path:
    """Resolve database file path."""
    db_path = config['project']['db_path']
    p = Path(db_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
