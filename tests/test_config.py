"""Tests for config.py — AppConfig defaults and load/save roundtrip."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import AppConfig, LLMConfig, StockfishConfig, load_config, save_config


def describe_StockfishConfig():
    def it_has_sensible_defaults():
        cfg = StockfishConfig()
        assert cfg.depth == 18
        assert cfg.threads == 2
        assert cfg.hash_mb == 256
        assert cfg.num_top_moves == 3

    def it_rejects_depth_below_1():
        with pytest.raises(Exception):
            StockfishConfig(depth=0)

    def it_rejects_depth_above_30():
        with pytest.raises(Exception):
            StockfishConfig(depth=31)


def describe_LLMConfig():
    def it_defaults_to_stub_provider():
        cfg = LLMConfig()
        assert cfg.provider == "stub"
        assert cfg.model == "stub-v1"
        assert cfg.api_key == ""

    def it_defaults_temperature_to_07():
        cfg = LLMConfig()
        assert cfg.temperature == 0.7


def describe_AppConfig():
    def it_defaults_player_name_to_Player():
        cfg = AppConfig()
        assert cfg.player_name == "Player"

    def it_defaults_chesscom_username_to_empty():
        cfg = AppConfig()
        assert cfg.chesscom_username == ""

    def it_composes_stockfish_and_llm_with_defaults():
        cfg = AppConfig()
        assert isinstance(cfg.stockfish, StockfishConfig)
        assert isinstance(cfg.llm, LLMConfig)


def describe_load_config():
    def it_returns_defaults_when_no_config_file_exists(tmp_path, monkeypatch):
        monkeypatch.setattr("config.CONFIG_PATH", tmp_path / "config.json")
        cfg = load_config()
        assert cfg.player_name == "Player"
        assert cfg.llm.provider == "stub"

    def it_merges_file_values_over_defaults(tmp_path, monkeypatch):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"player_name": "Fernando", "chesscom_username": "ferparra"}))
        monkeypatch.setattr("config.CONFIG_PATH", config_path)
        cfg = load_config()
        assert cfg.player_name == "Fernando"
        assert cfg.chesscom_username == "ferparra"


def describe_save_config():
    def it_writes_a_valid_json_file(tmp_path, monkeypatch):
        config_path = tmp_path / "config.json"
        monkeypatch.setattr("config.CONFIG_PATH", config_path)
        monkeypatch.setattr("config.DATA_DIR", tmp_path)
        cfg = AppConfig(player_name="TestUser")
        save_config(cfg)
        assert config_path.exists()
        raw = json.loads(config_path.read_text())
        assert raw["player_name"] == "TestUser"

    def it_roundtrips_through_load_and_save(tmp_path, monkeypatch):
        config_path = tmp_path / "config.json"
        monkeypatch.setattr("config.CONFIG_PATH", config_path)
        monkeypatch.setattr("config.DATA_DIR", tmp_path)

        original = AppConfig(
            player_name="Fernando",
            chesscom_username="ferparra",
            llm=LLMConfig(provider="openrouter", model="anthropic/claude-haiku-4-5-20251001"),
        )
        save_config(original)
        restored = load_config()

        assert restored.player_name == "Fernando"
        assert restored.chesscom_username == "ferparra"
        assert restored.llm.provider == "openrouter"
        assert restored.llm.model == "anthropic/claude-haiku-4-5-20251001"
