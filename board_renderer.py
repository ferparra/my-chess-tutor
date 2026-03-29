from __future__ import annotations

import chess
import chess.svg

from models import CriticalMoment, GameReview

# ---------------------------------------------------------------------------
# Arrow colour scheme
# ---------------------------------------------------------------------------

# The move that was actually played (the mistake) — shown in muted red
_COLOUR_PLAYED = "#c62828"

# The engine's recommended move — shown in green
_COLOUR_BEST = "#2e7d32"

# Square highlight for the piece that moved (light tint so arrows stay readable)
_FILL_FROM = "#ffd54f40"  # amber, semi-transparent


def render_moment(moment: CriticalMoment) -> str:
    """
    Render a critical moment as an SVG chess board string (no file I/O).

    Two arrows are drawn on the board position *before* the move:
      - Red  →  the move that was played (the mistake)
      - Green →  the engine's recommended move

    The board is oriented from the perspective of the player who moved
    (derived from the side-to-move in the FEN).

    Returns the SVG string, or "" if rendering failed.
    """
    if not moment.fen:
        return ""

    try:
        board = chess.Board(moment.fen)
    except ValueError:
        return ""

    # Side-to-move in the FEN = the player who made this move
    player_is_black = board.turn == chess.BLACK
    arrows: list[chess.svg.Arrow] = []
    fill: dict[int, str] = {}

    # --- Arrow for the move that was played ---
    played_move: chess.Move | None = None
    try:
        played_move = board.parse_san(moment.san)
        arrows.append(
            chess.svg.Arrow(played_move.from_square, played_move.to_square, color=_COLOUR_PLAYED)
        )
        fill[played_move.from_square] = _FILL_FROM
    except ValueError:
        pass  # unrecognised SAN — board still renders without the played-move arrow

    # --- Arrow for the recommended move ---
    try:
        best_move = board.parse_san(moment.best_move_san)
        if best_move != played_move:
            arrows.append(
                chess.svg.Arrow(best_move.from_square, best_move.to_square, color=_COLOUR_BEST)
            )
    except ValueError:
        pass  # unrecognised best_move_san — board still renders without the recommendation arrow

    return chess.svg.board(
        board,
        arrows=arrows,
        fill=fill,
        size=480,
        coordinates=True,
        flipped=player_is_black,
    )


def embed_review_images(review: GameReview) -> GameReview:
    """
    Render SVG boards for every critical moment and embed them in-place.

    Mutates each CriticalMoment.svg field directly on the review object and
    returns the same review for convenience.  No files are written to disk.
    """
    for moment in review.critical_moments:
        svg = render_moment(moment)
        if svg:
            moment.svg = svg
    return review
