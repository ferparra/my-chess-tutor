"""Tests for models.py — domain types and their defaults."""
from __future__ import annotations

import pytest
from datetime import datetime

from models import (
    CriticalMoment,
    EngineEval,
    GameAnalysis,
    GameMetadata,
    GameResult,
    GameReview,
    MoveClassification,
    PlayerColor,
    PlayerProfile,
    PlayerStats,
)


def describe_MoveClassification():
    def it_exposes_all_five_levels():
        levels = {c.value for c in MoveClassification}
        assert levels == {"brilliant", "good", "inaccuracy", "mistake", "blunder"}

    def it_is_a_string_enum():
        assert MoveClassification.BRILLIANT == "brilliant"
        assert MoveClassification.BLUNDER == "blunder"


def describe_PlayerColor():
    def it_has_white_and_black():
        assert PlayerColor.WHITE == "white"
        assert PlayerColor.BLACK == "black"


def describe_GameResult():
    def it_uses_pgn_notation():
        assert GameResult.WHITE_WINS == "1-0"
        assert GameResult.BLACK_WINS == "0-1"
        assert GameResult.DRAW == "1/2-1/2"
        assert GameResult.UNKNOWN == "*"


def describe_EngineEval():
    def it_defaults_centipawns_and_mate_in_to_none():
        ev = EngineEval(depth=18, best_uci="e2e4", best_san="e4")
        assert ev.centipawns is None
        assert ev.mate_in is None

    def it_defaults_pv_to_empty_list():
        ev = EngineEval(depth=18, best_uci="e2e4", best_san="e4")
        assert ev.pv == []

    def it_stores_mate_in():
        ev = EngineEval(depth=18, best_uci="e2e4", best_san="e4", mate_in=2)
        assert ev.mate_in == 2


def describe_GameMetadata():
    def it_defaults_players_to_question_mark():
        m = GameMetadata()
        assert m.white_player == "?"
        assert m.black_player == "?"

    def it_defaults_result_to_unknown():
        m = GameMetadata()
        assert m.result == GameResult.UNKNOWN


def describe_CriticalMoment():
    def it_defaults_svg_to_empty_string():
        m = CriticalMoment(
            move_number=1,
            san="e4",
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            classification=MoveClassification.GOOD,
            principle="Center Control",
            what_happened="Good central control.",
            what_to_ask="Am I fighting for the center?",
            best_move_san="e4",
        )
        assert m.svg == ""

    def it_accepts_an_svg_string():
        m = CriticalMoment(
            move_number=1,
            san="e4",
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            classification=MoveClassification.GOOD,
            principle="Center Control",
            what_happened="Took the center.",
            what_to_ask="Am I fighting for the center?",
            best_move_san="e4",
            svg="<svg/>",
        )
        assert m.svg == "<svg/>"


def describe_GameAnalysis():
    def it_initialises_all_stats_to_zero():
        ga = GameAnalysis(
            metadata=GameMetadata(),
            player_color=PlayerColor.WHITE,
        )
        assert ga.blunders == 0
        assert ga.mistakes == 0
        assert ga.inaccuracies == 0
        assert ga.total_moves == 0
        assert ga.average_cp_loss == 0.0
        assert ga.accuracy_pct == 0.0

    def it_generates_a_unique_game_id():
        ga1 = GameAnalysis(metadata=GameMetadata(), player_color=PlayerColor.WHITE)
        ga2 = GameAnalysis(metadata=GameMetadata(), player_color=PlayerColor.WHITE)
        assert ga1.game_id != ga2.game_id

    def it_initialises_move_analyses_to_empty_list():
        ga = GameAnalysis(metadata=GameMetadata(), player_color=PlayerColor.WHITE)
        assert ga.move_analyses == []


def describe_PlayerProfile():
    def it_defaults_level_to_beginner():
        p = PlayerProfile(player_name="Alice")
        assert p.estimated_level == "beginner"

    def it_defaults_game_ids_to_empty_list():
        p = PlayerProfile(player_name="Alice")
        assert p.game_ids == []


def describe_PlayerStats():
    def it_initialises_all_counters_to_zero():
        s = PlayerStats()
        assert s.games_analyzed == 0
        assert s.total_blunders == 0
        assert s.total_mistakes == 0
        assert s.total_inaccuracies == 0
        assert s.average_cp_loss_all == 0.0
        assert s.average_accuracy_all == 0.0

    def it_initialises_blunder_themes_to_empty_dict():
        s = PlayerStats()
        assert s.blunder_themes == {}
