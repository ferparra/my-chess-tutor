from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MoveClassification(str, Enum):
    BRILLIANT = "brilliant"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    BLUNDER = "blunder"


class PlayerColor(str, Enum):
    WHITE = "white"
    BLACK = "black"


class GameResult(str, Enum):
    WHITE_WINS = "1-0"
    BLACK_WINS = "0-1"
    DRAW = "1/2-1/2"
    UNKNOWN = "*"


class Move(BaseModel):
    uci: str
    san: str
    move_number: int
    color: PlayerColor
    fen_before: str


class EngineEval(BaseModel):
    centipawns: Optional[int] = None
    mate_in: Optional[int] = None
    depth: int
    best_uci: str
    best_san: str
    pv: list[str] = Field(default_factory=list)


class MoveAnalysis(BaseModel):
    move: Move
    played_eval: EngineEval
    best_eval: EngineEval
    cp_loss: int
    classification: MoveClassification
    missed_tactic: bool = False
    comment: str = ""


class GameMetadata(BaseModel):
    white_player: str = "?"
    black_player: str = "?"
    date: Optional[str] = None
    event: Optional[str] = None
    result: GameResult = GameResult.UNKNOWN
    opening_name: Optional[str] = None
    pgn_source: str = ""


class GameAnalysis(BaseModel):
    game_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: GameMetadata
    player_color: PlayerColor
    move_analyses: list[MoveAnalysis] = Field(default_factory=list)
    total_moves: int = 0
    blunders: int = 0
    mistakes: int = 0
    inaccuracies: int = 0
    average_cp_loss: float = 0.0
    accuracy_pct: float = 0.0


class CriticalMoment(BaseModel):
    """A pivotal move in the game, enriched for board image rendering."""
    move_number: int
    san: str
    fen: str                    # position BEFORE the move — for chess board rendering
    classification: MoveClassification
    principle: str              # the chess principle at stake, e.g. "King Safety"
    what_happened: str          # plain-English explanation tied to the principle
    what_to_ask: str            # the self-check question to internalise the principle
    best_move_san: str
    svg: str = ""               # SVG board diagram with arrows, embedded for upstream use


class GameReview(BaseModel):
    game_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    opening_comment: str        # phase-level feedback on development / king safety / center
    summary: str
    critical_moments: list[CriticalMoment]
    principles_to_study: list[str]  # 1-3 fundamental principles to focus on next
    weekly_exercise: str        # one concrete practice task
    encouragement: str


class GameRecord(BaseModel):
    analysis: GameAnalysis
    review: Optional[GameReview] = None
    pgn: str = ""


class PlayerStats(BaseModel):
    games_analyzed: int = 0
    total_blunders: int = 0
    total_mistakes: int = 0
    total_inaccuracies: int = 0
    average_cp_loss_all: float = 0.0
    average_accuracy_all: float = 0.0
    blunder_themes: dict[str, int] = Field(default_factory=dict)


class PlayerProfile(BaseModel):
    player_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    estimated_level: str = "beginner"
    stats: PlayerStats = Field(default_factory=PlayerStats)
    game_ids: list[str] = Field(default_factory=list)
