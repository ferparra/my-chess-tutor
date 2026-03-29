"""Tests for fetcher.py — ChessComGame parsing and HTTP error handling."""
from __future__ import annotations

import json
from typing import Any
import pytest
from unittest.mock import MagicMock

from fetcher import ChessComError, ChessComFetcher, ChessComGame, ChessComPlayer
from models import PlayerColor


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_SAMPLE_PGN = """\
[Event "Live Chess"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[Date "2024.03.15"]
[Opening "Ruy Lopez"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 1-0
"""

_SAMPLE_GAME_DICT: dict[str, Any] = {
    "url": "https://chess.com/game/1",
    "pgn": _SAMPLE_PGN,
    "time_class": "rapid",
    "time_control": "600",
    "rated": True,
    "white": {"username": "alice", "rating": 1200, "result": "win", "uuid": "w-uuid"},
    "black": {"username": "bob", "rating": 1100, "result": "checkmated", "uuid": "b-uuid"},
    "uuid": "game-uuid",
    "end_time": 1_710_000_000,
}


@pytest.fixture
def game() -> ChessComGame:
    return ChessComGame.model_validate(_SAMPLE_GAME_DICT)


# ---------------------------------------------------------------------------
# ChessComGame model parsing
# ---------------------------------------------------------------------------


def describe_ChessComGame():
    def it_parses_from_api_dict(game):
        assert game.url == "https://chess.com/game/1"
        assert game.time_class == "rapid"
        assert game.rated is True

    def it_strips_unknown_api_fields():
        raw = dict(_SAMPLE_GAME_DICT)
        raw["ECOUrl"] = "https://chess.com/eco/foo"
        raw["tcn"] = "some-tcn"
        raw["fen"] = "some-fen"
        raw["initial_setup"] = "setup"
        raw["rules"] = "chess"
        game = ChessComGame.model_validate(raw)
        assert game.url == raw["url"]  # still parsed fine

    def it_defaults_accuracies_to_none(game):
        assert game.accuracies is None

    def describe_player_color():
        def it_returns_white_for_white_username(game):
            assert game.player_color("alice") == PlayerColor.WHITE

        def it_returns_black_for_black_username(game):
            assert game.player_color("bob") == PlayerColor.BLACK

        def it_is_case_insensitive(game):
            assert game.player_color("Alice") == PlayerColor.WHITE
            assert game.player_color("BOB") == PlayerColor.BLACK

    def describe_result_for():
        def it_returns_win_for_winner(game):
            assert game.result_for("alice") == "win"

        def it_returns_loss_for_loser(game):
            assert game.result_for("bob") == "loss"

        def it_returns_draw_for_agreed_draw():
            raw: dict[str, Any] = dict(_SAMPLE_GAME_DICT)
            raw["white"] = {**_SAMPLE_GAME_DICT["white"], "result": "agreed"}
            raw["black"] = {**_SAMPLE_GAME_DICT["black"], "result": "agreed"}
            g = ChessComGame.model_validate(raw)
            assert g.result_for("alice") == "draw"

        def it_returns_draw_for_stalemate():
            raw: dict[str, Any] = dict(_SAMPLE_GAME_DICT)
            raw["white"] = {**_SAMPLE_GAME_DICT["white"], "result": "stalemate"}
            raw["black"] = {**_SAMPLE_GAME_DICT["black"], "result": "stalemate"}
            g = ChessComGame.model_validate(raw)
            assert g.result_for("alice") == "draw"

    def describe_opponent():
        def it_returns_black_player_when_user_is_white(game):
            opp = game.opponent("alice")
            assert opp.username == "bob"

        def it_returns_white_player_when_user_is_black(game):
            opp = game.opponent("bob")
            assert opp.username == "alice"

    def describe_opening_name():
        def it_parses_opening_from_pgn_header(game):
            assert game.opening_name() == "Ruy Lopez"

        def it_returns_none_when_opening_header_is_absent():
            raw = dict(_SAMPLE_GAME_DICT)
            raw["pgn"] = "[White \"alice\"]\n\n1. e4 1-0\n"
            g = ChessComGame.model_validate(raw)
            assert g.opening_name() is None

    def describe_pgn_date():
        def it_parses_date_from_pgn_header(game):
            assert game.pgn_date() == "2024.03.15"

        def it_returns_none_when_date_header_is_absent():
            raw = dict(_SAMPLE_GAME_DICT)
            raw["pgn"] = "[White \"alice\"]\n\n1. e4 1-0\n"
            g = ChessComGame.model_validate(raw)
            assert g.pgn_date() is None

    def describe_end_datetime():
        def it_returns_a_utc_datetime(game):
            from datetime import timezone
            dt = game.end_datetime()
            assert dt.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# ChessComFetcher HTTP behaviour — no real network calls
# ---------------------------------------------------------------------------


def describe_ChessComFetcher():
    def describe__get():
        def it_raises_on_404(mocker):
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_resp.ok = False
            mocker.patch("fetcher.requests.Session.get", return_value=mock_resp)
            with ChessComFetcher() as fetcher:
                with pytest.raises(ChessComError, match="Not found"):
                    fetcher._get("https://api.chess.com/pub/player/unknown")

        def it_raises_on_non_ok_status(mocker):
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.ok = False
            mocker.patch("fetcher.requests.Session.get", return_value=mock_resp)
            with ChessComFetcher() as fetcher:
                with pytest.raises(ChessComError, match="HTTP 500"):
                    fetcher._get("https://api.chess.com/pub/player/alice")

        def it_returns_json_on_success(mocker):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.ok = True
            mock_resp.json.return_value = {"archives": ["url1", "url2"]}
            mocker.patch("fetcher.requests.Session.get", return_value=mock_resp)
            with ChessComFetcher() as fetcher:
                result = fetcher._get("https://api.chess.com/pub/player/alice/games/archives")
            assert result == {"archives": ["url1", "url2"]}

    def describe_get_archives():
        def it_returns_list_of_archive_urls(mocker):
            archives = ["https://api.chess.com/pub/player/alice/games/2024/01"]
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.ok = True
            mock_resp.json.return_value = {"archives": archives}
            mocker.patch("fetcher.requests.Session.get", return_value=mock_resp)
            with ChessComFetcher() as fetcher:
                result = fetcher.get_archives("alice")
            assert result == archives

        def it_returns_empty_list_when_no_archives_key(mocker):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.ok = True
            mock_resp.json.return_value = {}
            mocker.patch("fetcher.requests.Session.get", return_value=mock_resp)
            with ChessComFetcher() as fetcher:
                result = fetcher.get_archives("alice")
            assert result == []

    def describe_get_monthly_games():
        def it_parses_valid_games(mocker):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.ok = True
            mock_resp.json.return_value = {"games": [_SAMPLE_GAME_DICT]}
            mocker.patch("fetcher.requests.Session.get", return_value=mock_resp)
            with ChessComFetcher() as fetcher:
                games = fetcher.get_monthly_games("https://api.chess.com/pub/player/alice/games/2024/03")
            assert len(games) == 1
            assert games[0].time_class == "rapid"

        def it_skips_malformed_game_entries(mocker):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.ok = True
            mock_resp.json.return_value = {"games": [{"bad": "data"}, _SAMPLE_GAME_DICT]}
            mocker.patch("fetcher.requests.Session.get", return_value=mock_resp)
            with ChessComFetcher() as fetcher:
                games = fetcher.get_monthly_games("https://api.chess.com/...")
            assert len(games) == 1
