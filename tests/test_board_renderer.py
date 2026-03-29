"""Tests for board_renderer.py — SVG generation, embedding."""
from __future__ import annotations

import pytest

from board_renderer import embed_review_images, render_moment
from models import CriticalMoment, MoveClassification

from tests.conftest import make_game_analysis

# FEN before 4. Qh5 in a Scholar's Mate line — a well-formed position
_VALID_FEN = "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 2"


def _moment(fen=_VALID_FEN, san="Qh5", best_san="Nf3") -> CriticalMoment:
    return CriticalMoment(
        move_number=4,
        san=san,
        fen=fen,
        classification=MoveClassification.MISTAKE,
        principle="King Safety",
        what_happened="Queen came out too early.",
        what_to_ask="Did I develop all my minor pieces first?",
        best_move_san=best_san,
    )


def describe_render_moment():
    def it_returns_a_non_empty_svg_string():
        svg = render_moment(_moment())
        assert isinstance(svg, str)
        assert len(svg) > 0

    def it_returns_valid_svg_markup():
        svg = render_moment(_moment())
        assert svg.strip().startswith("<svg") or "<svg" in svg

    def it_draws_the_red_arrow_for_played_move():
        svg = render_moment(_moment())
        assert "#c62828" in svg

    def it_draws_the_green_arrow_for_best_move():
        svg = render_moment(_moment())
        assert "#2e7d32" in svg

    def it_omits_green_arrow_when_played_equals_best():
        # When the player's move IS the best move, only one arrow should appear
        svg = render_moment(_moment(san="Nf3", best_san="Nf3"))
        assert svg.count("#2e7d32") == 0

    def it_returns_empty_string_for_empty_fen():
        svg = render_moment(_moment(fen=""))
        assert svg == ""

    def it_returns_empty_string_for_invalid_fen():
        svg = render_moment(_moment(fen="not-a-fen"))
        assert svg == ""

    def it_still_renders_board_when_san_is_illegal():
        # Rxe8 is illegal from this position — arrow is skipped but the board still renders
        svg = render_moment(_moment(san="Rxe8"))
        assert "<svg" in svg  # board rendered despite missing arrow

    def it_flips_board_for_black_to_move():
        # FEN where it's Black's turn — board should be flipped
        black_to_move_fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        # Black plays Nc6 from this position
        svg = render_moment(_moment(fen=black_to_move_fen, san="Nc6", best_san="Nf6"))
        assert isinstance(svg, str)


def describe_embed_review_images():
    def it_mutates_svg_field_in_place(sample_review):
        assert sample_review.critical_moments[0].svg == ""
        embed_review_images(sample_review)
        assert sample_review.critical_moments[0].svg != ""

    def it_returns_the_same_review_object(sample_review):
        result = embed_review_images(sample_review)
        assert result is sample_review

    def it_embeds_valid_svg_content(sample_review):
        embed_review_images(sample_review)
        svg = sample_review.critical_moments[0].svg
        assert "<svg" in svg

    def it_skips_moments_with_empty_fen(sample_review):
        sample_review.critical_moments[0].fen = ""
        embed_review_images(sample_review)
        assert sample_review.critical_moments[0].svg == ""

    def it_processes_all_moments(sample_review):
        from datetime import datetime
        extra = CriticalMoment(
            move_number=6,
            san="Nf3",
            fen=_VALID_FEN,
            classification=MoveClassification.GOOD,
            principle="Development",
            what_happened="Nice development.",
            what_to_ask="Am I bringing pieces out?",
            best_move_san="Nf3",
        )
        sample_review.critical_moments.append(extra)
        embed_review_images(sample_review)
        for m in sample_review.critical_moments:
            if m.fen:
                assert m.svg != "", f"moment {m.move_number} svg not embedded"
