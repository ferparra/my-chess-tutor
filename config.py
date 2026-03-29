from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DATA_DIR = Path.home() / ".my_chess_tutor"
CONFIG_PATH = DATA_DIR / "config.json"


class StockfishConfig(BaseModel):
    binary_path: Path = DATA_DIR / "stockfish"
    depth: int = Field(default=18, ge=1, le=30)
    threads: int = Field(default=2, ge=1)
    hash_mb: int = Field(default=256, ge=16)
    num_top_moves: int = Field(default=3, ge=1, le=5)


class LLMConfig(BaseModel):
    provider: str = "stub"
    model: str = "stub-v1"
    api_key: str = ""
    max_tokens: int = 512
    temperature: float = 0.7


class AppConfig(BaseModel):
    stockfish: StockfishConfig = Field(default_factory=StockfishConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    player_name: str = "Player"
    chesscom_username: str = ""
    data_dir: Path = DATA_DIR


def load_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        return AppConfig()
    raw: dict[str, Any] = json.loads(CONFIG_PATH.read_text())
    return AppConfig.model_validate(raw)


def save_config(config: AppConfig) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(config.model_dump_json(indent=2))
