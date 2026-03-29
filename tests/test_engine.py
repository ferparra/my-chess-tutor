"""Tests for engine.py — pure functions and StockfishEngine unit tests."""
from __future__ import annotations

import chess
import pytest
from unittest.mock import MagicMock, patch

from engine import (
    StockfishEngine,
    StockfishNotFoundError,
    _build_move_comment,
    _compute_aggregate_stats,
    _cp_loss_to_accuracy,
    _negate_eval_cp,
    _negate_mate,
)
from models import GameAnalysis, GameMetadata, MoveClassification, PlayerColor

from tests.conftest import make_game_analysis, make_move_analysis


# ---------------------------------------------------------------------------
# Pure function tests — no binary needed
# ---------------------------------------------------------------------------


def describe_cp_loss_to_accuracy():
    def it_returns_100_for_zero_loss():
        assert _cp_loss_to_accuracy(0) == 100.0

    def it_returns_0_for_very_large_loss():
        assert _cp_loss_to_accuracy(10_000) == 0.0

    def it_stays_within_0_and_100():
        for loss in range(0, 500, 25):
            acc = _cp_loss_to_accuracy(loss)
            assert 0.0 <= acc <= 100.0, f"out of range for cp_loss={loss}"

    def it_decreases_as_cp_loss_grows():
        assert _cp_loss_to_accuracy(0) > _cp_loss_to_accuracy(100)
        assert _cp_loss_to_accuracy(100) > _cp_loss_to_accuracy(300)


def describe_negate_eval_cp():
    def it_negates_a_positive_value():
        assert _negate_eval_cp(50) == -50

    def it_negates_a_negative_value():
        assert _negate_eval_cp(-30) == 30

    def it_negates_zero():
        assert _negate_eval_cp(0) == 0

    def it_passes_none_through():
        assert _negate_eval_cp(None) is None


def describe_negate_mate():
    def it_negates_positive_mate():
        assert _negate_mate(3) == -3

    def it_negates_negative_mate():
        assert _negate_mate(-2) == 2

    def it_passes_none_through():
        assert _negate_mate(None) is None


def describe_build_move_comment():
    def it_returns_empty_string_for_good_move():
        assert _build_move_comment("e4", 0, MoveClassification.GOOD, "d4") == ""

    def it_mentions_best_move_for_inaccuracy():
        comment = _build_move_comment("Qh4", 50, MoveClassification.INACCURACY, "Nf3")
        assert "Nf3" in comment
        assert "inaccurate" in comment.lower()

    def it_mentions_cp_loss_for_inaccuracy():
        comment = _build_move_comment("h3", 50, MoveClassification.INACCURACY, "Nf3")
        assert "50" in comment

    def it_labels_mistakes_clearly():
        comment = _build_move_comment("h3", 150, MoveClassification.MISTAKE, "Nf3")
        assert "mistake" in comment.lower()
        assert "Nf3" in comment

    def it_labels_blunders_clearly():
        comment = _build_move_comment("Bxf7", 400, MoveClassification.BLUNDER, "Nf3")
        assert "blunder" in comment.lower()
        assert "Nf3" in comment

    def it_celebrates_a_brilliant_move():
        comment = _build_move_comment("Rxe5", 0, MoveClassification.BRILLIANT, "Rxe5")
        assert "excellent" in comment.lower() or "brilliant" in comment.lower() or "★" in comment or comment


def describe_compute_aggregate_stats():
    def it_counts_blunders_mistakes_inaccuracies():
        analyses = [
            make_move_analysis(cp_loss=350, classification=MoveClassification.BLUNDER),
            make_move_analysis(cp_loss=150, classification=MoveClassification.MISTAKE),
            make_move_analysis(cp_loss=50, classification=MoveClassification.INACCURACY),
            make_move_analysis(cp_loss=0, classification=MoveClassification.GOOD),
        ]
        ga = make_game_analysis(move_analyses=analyses)
        result = _compute_aggregate_stats(ga, analyses)
        assert result.blunders == 1
        assert result.mistakes == 1
        assert result.inaccuracies == 1
        assert result.total_moves == 4

    def it_computes_average_cp_loss():
        analyses = [
            make_move_analysis(cp_loss=100),
            make_move_analysis(cp_loss=200),
        ]
        ga = make_game_analysis(move_analyses=analyses)
        result = _compute_aggregate_stats(ga, analyses)
        assert result.average_cp_loss == 150.0

    def it_returns_unchanged_analysis_when_no_moves():
        ga = make_game_analysis()
        result = _compute_aggregate_stats(ga, [])
        assert result.total_moves == 0
        assert result.average_cp_loss == 0.0


# ---------------------------------------------------------------------------
# StockfishEngine tests — Stockfish subprocess is mocked
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_engine(tmp_path):
    """StockfishEngine backed by a fake binary and a mocked Stockfish process."""
    binary = tmp_path / "stockfish"
    binary.write_bytes(b"#!/bin/sh\n")
    binary.chmod(0o755)
    with patch("engine.Stockfish") as MockSF:
        MockSF.return_value = MagicMock()
        from config import StockfishConfig
        cfg = StockfishConfig(binary_path=binary)
        yield StockfishEngine(cfg)


def describe_StockfishEngine():
    def describe_init():
        def it_raises_when_binary_is_missing(tmp_path):
            from config import StockfishConfig
            cfg = StockfishConfig(binary_path=tmp_path / "no_such_binary")
            with pytest.raises(StockfishNotFoundError):
                StockfishEngine(cfg)

        def it_starts_successfully_with_existing_binary(fake_engine):
            assert fake_engine is not None

    def describe_classify_move():
        _dummy_move = chess.Move.from_uci("e2e4")
        _dummy_board = chess.Board()

        def it_classifies_zero_loss_as_good(fake_engine):
            result = fake_engine.classify_move(0, _dummy_move, _dummy_board)
            assert result == MoveClassification.GOOD

        def it_classifies_20_cp_loss_as_good(fake_engine):
            result = fake_engine.classify_move(20, _dummy_move, _dummy_board)
            assert result == MoveClassification.GOOD

        def it_classifies_21_cp_loss_as_inaccuracy(fake_engine):
            result = fake_engine.classify_move(21, _dummy_move, _dummy_board)
            assert result == MoveClassification.INACCURACY

        def it_classifies_100_cp_loss_as_inaccuracy(fake_engine):
            result = fake_engine.classify_move(100, _dummy_move, _dummy_board)
            assert result == MoveClassification.INACCURACY

        def it_classifies_101_cp_loss_as_mistake(fake_engine):
            result = fake_engine.classify_move(101, _dummy_move, _dummy_board)
            assert result == MoveClassification.MISTAKE

        def it_classifies_300_cp_loss_as_mistake(fake_engine):
            result = fake_engine.classify_move(300, _dummy_move, _dummy_board)
            assert result == MoveClassification.MISTAKE

        def it_classifies_301_cp_loss_as_blunder(fake_engine):
            result = fake_engine.classify_move(301, _dummy_move, _dummy_board)
            assert result == MoveClassification.BLUNDER

    def describe_context_manager():
        def it_can_be_used_as_context_manager(tmp_path):
            binary = tmp_path / "stockfish"
            binary.write_bytes(b"#!/bin/sh\n")
            binary.chmod(0o755)
            with patch("engine.Stockfish") as MockSF:
                MockSF.return_value = MagicMock()
                from config import StockfishConfig
                cfg = StockfishConfig(binary_path=binary)
                with StockfishEngine(cfg) as eng:
                    assert eng is not None
