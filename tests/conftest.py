"""Shared fixtures and builder helpers for all test modules."""
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
    Move,
    MoveAnalysis,
    MoveClassification,
    PlayerColor,
    PlayerProfile,
    PlayerStats,
)


# ---------------------------------------------------------------------------
# Lightweight builder helpers (not fixtures — call them directly in tests)
# ---------------------------------------------------------------------------


def make_eval(best_san: str = "e4", centipawns: int = 50) -> EngineEval:
    return EngineEval(depth=18, best_uci="e2e4", best_san=best_san, centipawns=centipawns)


def make_move(
    san: str = "e4",
    move_number: int = 1,
    color: PlayerColor = PlayerColor.WHITE,
    fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
) -> Move:
    return Move(uci="e2e4", san=san, move_number=move_number, color=color, fen_before=fen)


def make_move_analysis(
    cp_loss: int = 0,
    classification: MoveClassification = MoveClassification.GOOD,
    color: PlayerColor = PlayerColor.WHITE,
    move_number: int = 1,
    san: str = "e4",
    comment: str = "",
    best_san: str = "d4",
) -> MoveAnalysis:
    move = make_move(san=san, move_number=move_number, color=color)
    eval_ = make_eval(best_san=best_san)
    return MoveAnalysis(
        move=move,
        played_eval=eval_,
        best_eval=eval_,
        cp_loss=cp_loss,
        classification=classification,
        comment=comment,
    )


def make_game_analysis(
    player_color: PlayerColor = PlayerColor.WHITE,
    move_analyses: list[MoveAnalysis] | None = None,
    blunders: int = 0,
    mistakes: int = 0,
    inaccuracies: int = 0,
    accuracy_pct: float = 70.0,
    opening_name: str | None = None,
) -> GameAnalysis:
    return GameAnalysis(
        metadata=GameMetadata(
            white_player="Alice",
            black_player="Bob",
            result=GameResult.WHITE_WINS,
            opening_name=opening_name,
        ),
        player_color=player_color,
        move_analyses=move_analyses or [],
        blunders=blunders,
        mistakes=mistakes,
        inaccuracies=inaccuracies,
        accuracy_pct=accuracy_pct,
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def profile() -> PlayerProfile:
    return PlayerProfile(
        player_name="TestPlayer",
        estimated_level="beginner",
        stats=PlayerStats(),
    )


@pytest.fixture
def scholars_mate_fen() -> str:
    """Position where 4. Qh5 is the Scholar's Mate threat — valid FEN for board rendering."""
    return "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 2"


@pytest.fixture
def minimal_moment(scholars_mate_fen) -> CriticalMoment:
    return CriticalMoment(
        move_number=4,
        san="Qh5",
        fen=scholars_mate_fen,
        classification=MoveClassification.MISTAKE,
        principle="King Safety",
        what_happened="Queen brought out too early.",
        what_to_ask="Have I developed all my minor pieces first?",
        best_move_san="Nf3",
    )


@pytest.fixture
def sample_review(minimal_moment) -> GameReview:
    return GameReview(
        game_id="test-game-id",
        generated_at=datetime.utcnow(),
        opening_comment="Decent opening.",
        summary="A learning game.",
        critical_moments=[minimal_moment],
        principles_to_study=["King Safety"],
        weekly_exercise="Castle before move 8.",
        encouragement="Keep it up!",
    )


# ---------------------------------------------------------------------------
# Sample PGN strings
# ---------------------------------------------------------------------------

SAMPLE_PGN = """\
[Event "Test"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]
[Date "2024.03.15"]
[Opening "Ruy Lopez"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 1-0
"""
