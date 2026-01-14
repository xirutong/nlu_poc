# settings.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import yaml


# =========================
# Project root
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent


# =========================
# Config dataclasses
# =========================

@dataclass(frozen=True)
class Paths:
    model_dir: Path


@dataclass(frozen=True)
class ASRConfig:
    model_name: str


@dataclass(frozen=True)
class NLUConfig:
    base_model_name: str


@dataclass(frozen=True)
class Settings:
    paths: Paths
    nlu: NLUConfig
    asr: ASRConfig


# =========================
# Loader
# =========================

def load_settings(
    config_path: str | Path = "config/default.yaml",
) -> Settings:
    """
    Load application settings.

    Priority:
        1. Environment variables
        2. YAML config file
    """

    # ---- load yaml ----
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise RuntimeError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # =========================
    # Paths
    # =========================

    model_dir_value = (
        os.environ.get("MODEL_DIR")
        or raw.get("paths", {}).get("model_dir")
    )

    if not model_dir_value:
        raise RuntimeError("MODEL_DIR is not set (env or config)")

    model_dir = Path(model_dir_value).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    paths = Paths(
        model_dir=model_dir,
    )

    # =========================
    # NLU
    # =========================

    try:
        nlu_raw = raw["nlu"]
        base_model_name = nlu_raw["base_model_name"]
    except KeyError as e:
        raise RuntimeError(f"Missing NLU config key: {e}")

    nlu = NLUConfig(
        base_model_name=base_model_name,
    )

    # =========================
    # ASR
    # =========================

    try:
        asr_raw = raw["asr"]
        model_name = asr_raw["model_name"]
    except KeyError as e:
        raise RuntimeError(f"Missing ASR config key: {e}")

    asr = ASRConfig(
        model_name=model_name,
    )

    # =========================
    # Final settings
    # =========================

    return Settings(
        paths=paths,
        nlu=nlu,
        asr=asr,
    )
