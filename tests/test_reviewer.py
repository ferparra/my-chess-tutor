"""Tests for reviewer.py — StubLLMProvider, ReviewGenerator logic."""
from __future__ import annotations

import pytest

from models import MoveClassification, PlayerColor
from reviewer import ReviewGenerator, StubLLMProvider, _LLMMoment, _LLMReview

from tests.conftest import make_game_analysis, make_move_analysis


def describe_StubLLMProvider():
    def it_is_always_available():
        assert StubLLMProvider().is_available() is True

    def it_returns_empty_string_from_complete():
        stub = StubLLMProvider()
        assert stub.complete("any prompt") == ""

    def it_raises_for_unknown_schemas():
        from pydantic import BaseModel
        class SomeOtherSchema(BaseModel):
            x: int

        stub = StubLLMProvider()
        with pytest.raises(TypeError):
            stub.complete_structured("prompt", "system", SomeOtherSchema)

    def it_returns_a_valid_llm_review_with_no_context():
        stub = StubLLMProvider()
        stub.set_context({})
        result = stub.complete_structured("prompt", "system", _LLMReview)
        assert isinstance(result, _LLMReview)
        assert result.opening_comment
        assert result.summary
        assert result.weekly_exercise
        assert result.encouragement

    def it_reflects_blunder_count_in_summary():
        stub = StubLLMProvider()
        stub.set_context({"blunders": 3, "accuracy": 55.0, "mistakes": 1, "inaccuracies": 0})
        result = stub.complete_structured("prompt", "system", _LLMReview)
        assert "3" in result.summary

    def it_builds_moments_from_context():
        stub = StubLLMProvider()
        stub.set_context({
            "moments": [
                {"move_number": 10, "san": "Bxf7", "comment": "bad bishop sac", "best_san": "Nf3"},
            ],
            "top_theme": "Tactics",
        })
        result = stub.complete_structured("prompt", "system", _LLMReview)
        assert len(result.critical_moments) == 1
        assert result.critical_moments[0].move_number == 10
        assert result.critical_moments[0].san == "Bxf7"


def describe_ReviewGenerator():
    @pytest.fixture
    def stub() -> StubLLMProvider:
        return StubLLMProvider()

    @pytest.fixture
    def generator(stub) -> ReviewGenerator:
        return ReviewGenerator(stub)

    def describe_select_critical_moves():
        def it_picks_only_blunders_and_mistakes(generator):
            analyses = [
                make_move_analysis(cp_loss=400, classification=MoveClassification.BLUNDER, san="Bxf7"),
                make_move_analysis(cp_loss=150, classification=MoveClassification.MISTAKE, san="h3"),
                make_move_analysis(cp_loss=50, classification=MoveClassification.INACCURACY, san="a3"),
                make_move_analysis(cp_loss=0, classification=MoveClassification.GOOD, san="e4"),
            ]
            ga = make_game_analysis(move_analyses=analyses)
            result = generator._select_critical_moves(ga)
            sans = [ma.move.san for ma in result]
            assert "Bxf7" in sans
            assert "h3" in sans
            assert "a3" not in sans
            assert "e4" not in sans

        def it_limits_to_three_worst(generator):
            analyses = [
                make_move_analysis(cp_loss=400, classification=MoveClassification.BLUNDER, san="m1"),
                make_move_analysis(cp_loss=380, classification=MoveClassification.BLUNDER, san="m2"),
                make_move_analysis(cp_loss=350, classification=MoveClassification.BLUNDER, san="m3"),
                make_move_analysis(cp_loss=320, classification=MoveClassification.BLUNDER, san="m4"),
            ]
            ga = make_game_analysis(move_analyses=analyses)
            result = generator._select_critical_moves(ga)
            assert len(result) == 3

        def it_sorts_by_cp_loss_descending(generator):
            analyses = [
                make_move_analysis(cp_loss=150, classification=MoveClassification.MISTAKE, san="small"),
                make_move_analysis(cp_loss=400, classification=MoveClassification.BLUNDER, san="large"),
            ]
            ga = make_game_analysis(move_analyses=analyses)
            result = generator._select_critical_moves(ga)
            assert result[0].move.san == "large"

        def it_only_includes_player_moves(generator):
            analyses = [
                make_move_analysis(
                    cp_loss=400, classification=MoveClassification.BLUNDER,
                    color=PlayerColor.BLACK, san="opponent_blunder",
                ),
                make_move_analysis(
                    cp_loss=100, classification=MoveClassification.INACCURACY,
                    color=PlayerColor.WHITE, san="my_inaccuracy",
                ),
            ]
            ga = make_game_analysis(player_color=PlayerColor.WHITE, move_analyses=analyses)
            result = generator._select_critical_moves(ga)
            # opponent's blunder is BLACK, player is WHITE — should not be included
            assert all(ma.move.color == PlayerColor.WHITE for ma in result)

    def describe_extract_themes():
        def it_returns_a_non_empty_list(generator):
            ga = make_game_analysis(blunders=1, mistakes=1)
            result = generator._extract_themes(ga)
            assert isinstance(result, list)
            assert len(result) >= 1

        def it_detects_tactics_from_comment(generator):
            analyses = [
                make_move_analysis(
                    cp_loss=400,
                    classification=MoveClassification.BLUNDER,
                    comment="missed a fork on f7",
                ),
            ]
            ga = make_game_analysis(move_analyses=analyses)
            result = generator._extract_themes(ga)
            assert "Tactics" in result

        def it_falls_back_to_tactics_when_no_keywords_match(generator):
            ga = make_game_analysis(blunders=0, mistakes=1)
            result = generator._extract_themes(ga)
            assert result  # non-empty

    def describe_enrich_moments():
        def it_injects_fen_from_game_analysis(generator):
            fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
            analyses = [
                make_move_analysis(
                    cp_loss=400,
                    classification=MoveClassification.BLUNDER,
                    move_number=5,
                    san="Bxf7",
                ),
            ]
            analyses[0].move.fen_before = fen
            ga = make_game_analysis(move_analyses=analyses)

            llm_moments = [
                _LLMMoment(
                    move_number=5,
                    san="Bxf7",
                    principle="Piece Safety",
                    what_happened="Lost the bishop.",
                    what_to_ask="Is this sacrifice sound?",
                    best_move_san="Nf3",
                )
            ]
            result = generator._enrich_moments(llm_moments, ga)
            assert result[0].fen == fen

        def it_injects_classification_from_game_analysis(generator):
            analyses = [
                make_move_analysis(
                    cp_loss=400,
                    classification=MoveClassification.BLUNDER,
                    move_number=5,
                    san="Bxf7",
                ),
            ]
            ga = make_game_analysis(move_analyses=analyses)
            llm_moments = [
                _LLMMoment(
                    move_number=5, san="Bxf7", principle="Piece Safety",
                    what_happened="Lost it.", what_to_ask="?", best_move_san="Nf3",
                )
            ]
            result = generator._enrich_moments(llm_moments, ga)
            assert result[0].classification == MoveClassification.BLUNDER

        def it_uses_empty_fen_when_move_not_found_in_analysis(generator):
            ga = make_game_analysis()
            llm_moments = [
                _LLMMoment(
                    move_number=99, san="Rxe8", principle="Tactics",
                    what_happened="Missed tactic.", what_to_ask="?", best_move_san="Rxe8",
                )
            ]
            result = generator._enrich_moments(llm_moments, ga)
            assert result[0].fen == ""

    def describe_generate():
        def it_returns_a_game_review_with_matching_game_id(generator, profile):
            ga = make_game_analysis(blunders=1, mistakes=1)
            ga.move_analyses = [
                make_move_analysis(
                    cp_loss=400, classification=MoveClassification.BLUNDER,
                    move_number=10, san="Bxf7", comment="hanging bishop",
                ),
            ]
            review = generator.generate(ga, profile)
            assert review.game_id == ga.game_id

        def it_includes_opening_comment_and_summary(generator, profile):
            ga = make_game_analysis()
            review = generator.generate(ga, profile)
            assert review.opening_comment
            assert review.summary

        def it_includes_principles_to_study(generator, profile):
            ga = make_game_analysis(blunders=2)
            review = generator.generate(ga, profile)
            assert isinstance(review.principles_to_study, list)
            assert len(review.principles_to_study) >= 1

        def it_includes_encouragement(generator, profile):
            ga = make_game_analysis()
            review = generator.generate(ga, profile)
            assert review.encouragement
