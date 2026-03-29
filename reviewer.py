from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar, cast

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from pydantic import BaseModel, Field

from models import (
    CriticalMoment,
    GameAnalysis,
    GameReview,
    MoveAnalysis,
    MoveClassification,
    PlayerProfile,
)

T = TypeVar("T", bound=BaseModel)

TEMPLATES_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    # Templates are plain-text LLM prompts (.j2), not HTML/XML — autoescape is not applicable.
    # select_autoescape enables escaping only for html/xml extensions; .j2 files are unaffected.
    autoescape=select_autoescape(enabled_extensions=("html", "xml")),
)

# ---------------------------------------------------------------------------
# LLM response schemas
# LLM fills in language/principle fields only — no FEN (engine-sourced).
# Pydantic v2 model_json_schema() produces the response_format JSON Schema.
# ---------------------------------------------------------------------------

_CHESS_PRINCIPLES = (
    "King Safety",
    "Development",
    "Center Control",
    "Piece Safety",
    "Piece Activity",
    "Tactical Awareness",
    "Pawn Structure",
    "Piece Coordination",
    "Endgame Principles",
)


class _LLMMoment(BaseModel):
    move_number: int = Field(description="Full-move number from the PGN")
    san: str = Field(description="SAN notation of the move played, e.g. Rxe5")
    principle: str = Field(
        description=(
            "The single chess principle being illustrated, chosen from: "
            + ", ".join(f'"{p}"' for p in _CHESS_PRINCIPLES)
        )
    )
    what_happened: str = Field(
        description=(
            "One to two sentences explaining WHY the move goes against that principle. "
            "Do not mention centipawns or engine evaluation. "
            "Explain the idea a club coach would give, e.g. "
            "'By bringing your queen out early you fell behind in development "
            "and gave your opponent time to build a strong center.'"
        )
    )
    what_to_ask: str = Field(
        description=(
            "The concrete self-check question the player should ask themselves "
            "before making this type of move in future, e.g. "
            "'Have I developed all my minor pieces before bringing out the queen?'"
        )
    )
    best_move_san: str = Field(
        description="SAN notation of the move the engine preferred"
    )


class _LLMReview(BaseModel):
    """
    Structured game review — schema sent to OpenRouter via response_format.
    Fields are ordered from broad context to specific moments to action.
    """

    opening_comment: str = Field(
        description=(
            "1-2 sentences on how the opening phase went in terms of development, "
            "king safety, and center control. Be concrete: mention if they castled "
            "in time, whether they controlled the center, etc."
        )
    )
    summary: str = Field(
        description=(
            "2-3 sentences summarising the game from a principles perspective. "
            "Mention the result and one or two broad themes without dwelling on individual moves."
        )
    )
    critical_moments: list[_LLMMoment] = Field(
        description=(
            "Up to 3 of the most instructive moments, chosen for teaching value. "
            "Order them by which principle is most fundamental to fix first."
        )
    )
    principles_to_study: list[str] = Field(
        description=(
            "1 to 3 fundamental chess principles this player should focus on next, "
            "drawn from the critical moments. Use the same principle names as above."
        )
    )
    weekly_exercise: str = Field(
        description=(
            "One specific, actionable practice exercise for this week. "
            "Be concrete: e.g. 'Solve 10 mate-in-one puzzles per day on chess.com' "
            "or 'Play 5 games where your only goal is to castle before move 8'."
        )
    )
    encouragement: str = Field(
        description=(
            "A warm, specific closing sentence that acknowledges something "
            "the player did well — even if small — and leaves them motivated."
        )
    )


# ---------------------------------------------------------------------------
# LLM provider interface
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str:
        ...

    def complete_structured(self, prompt: str, system: str, schema: type[T]) -> T:
        """
        Default: instruct the LLM to return JSON, strip fences, parse with Pydantic.
        Override for native structured-output (e.g. OpenRouter response_format).
        """
        raw = self.complete(
            prompt + "\n\nRespond with valid JSON matching the schema. No markdown fences.",
            system,
        )
        content = raw.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return schema.model_validate_json(content.strip())

    @abstractmethod
    def is_available(self) -> bool:
        ...


# ---------------------------------------------------------------------------
# Stub provider (offline, template-free — builds _LLMReview directly)
# ---------------------------------------------------------------------------

_PRINCIPLE_FOR_THEME: dict[str, str] = {
    "Hanging pieces": "Piece Safety",
    "Back rank weakness": "King Safety",
    "Tactics": "Tactical Awareness",
    "Endgame technique": "Endgame Principles",
    "Opening principles": "Development",
}

_WHAT_TO_ASK: dict[str, str] = {
    "Piece Safety": (
        "Before I move, can my opponent take any of my pieces for free?"
    ),
    "King Safety": (
        "Is my king safe? Have I castled, and are there open files pointing at my king?"
    ),
    "Tactical Awareness": (
        "Before I move, can my opponent check me, capture something, or fork two of my pieces?"
    ),
    "Endgame Principles": (
        "Is my king active and moving toward the center? Am I creating a passed pawn?"
    ),
    "Development": (
        "Have I developed all my minor pieces before starting an attack or bringing out my queen?"
    ),
    "Center Control": (
        "Do I control or contest the central squares e4, d4, e5, d5?"
    ),
    "Piece Activity": (
        "Am I placing my pieces on their best squares, where they control the most board?"
    ),
    "Pawn Structure": (
        "Will this move create a pawn weakness — isolated, doubled, or backward?"
    ),
    "Piece Coordination": (
        "Are my pieces working together, or are they getting in each other's way?"
    ),
}

_EXERCISES: dict[str, str] = {
    "Piece Safety": (
        "Play 5 slow games (15+10) and before every single move ask: "
        "'Can my opponent take any piece for free?' Stop immediately if the answer is yes."
    ),
    "King Safety": (
        "Play 5 games with one rule: castle before move 8. "
        "Do not start any attack until your king is castled."
    ),
    "Tactical Awareness": (
        "Solve 10 mate-in-one and 10 one-move-winning-capture puzzles per day on chess.com or Lichess "
        "for one week. Speed is less important than spotting the pattern."
    ),
    "Development": (
        "Play 5 games of 1.e4 or 1.d4 and count your moves. "
        "Make sure all four minor pieces are developed before you move any piece twice or push the queen out."
    ),
    "Endgame Principles": (
        "Study basic king-and-pawn endgames: practice the opposition (two kings face each other) "
        "and the 'square rule' for pawn races. Chess.com's endgame drills are perfect."
    ),
}


class StubLLMProvider(LLMProvider):
    def __init__(self) -> None:
        self._ctx: dict[str, Any] = {}

    def set_context(self, ctx: dict[str, Any]) -> None:
        self._ctx = ctx

    def complete(self, prompt: str, system: str = "") -> str:
        return ""

    def complete_structured(self, prompt: str, system: str, schema: type[T]) -> T:
        if schema is not _LLMReview:
            raise TypeError(f"StubLLMProvider only handles _LLMReview, got {schema}")

        ctx = self._ctx
        level: str = ctx.get("level", "beginner")
        blunders: int = ctx.get("blunders", 0)
        accuracy: float = ctx.get("accuracy", 0.0)
        mistakes: int = ctx.get("mistakes", 0)
        inaccuracies: int = ctx.get("inaccuracies", 0)
        top_theme: str = ctx.get("top_theme", "Tactics")
        opening_name: str = ctx.get("opening_name", "")

        top_principle = _PRINCIPLE_FOR_THEME.get(top_theme, "Tactical Awareness")
        exercise = _EXERCISES.get(top_principle, _EXERCISES["Tactical Awareness"])
        what_to_ask = _WHAT_TO_ASK.get(top_principle, _WHAT_TO_ASK["Tactical Awareness"])

        raw_moments: list[dict[str, Any]] = ctx.get("moments", [])
        llm_moments = [
            _LLMMoment(
                move_number=m["move_number"],
                san=m["san"],
                principle=top_principle,
                what_happened=m["comment"],
                what_to_ask=what_to_ask,
                best_move_san=m["best_san"],
            )
            for m in raw_moments
        ]

        quality = "tough" if blunders >= 2 else ("mixed" if blunders == 1 else "solid")
        opening_phrase = f"in the {opening_name}" if opening_name else "in the opening"
        opening_comment = (
            f"You showed some good instincts {opening_phrase}, "
            f"but {top_principle.lower()} was the main challenge this game."
        )
        summary = (
            f"A {quality} game with {accuracy:.0f}% accuracy — "
            f"{blunders} blunder(s), {mistakes} mistake(s), {inaccuracies} inaccuracy/inaccuracies. "
            f"The main pattern to work on is {top_principle.lower()}."
        )

        # StubLLMProvider only handles _LLMReview (enforced by the TypeError guard above).
        # cast<T> tells the type checker that the concrete _LLMReview satisfies the generic T.
        return cast(T, _LLMReview(
            opening_comment=opening_comment,
            summary=summary,
            critical_moments=llm_moments,
            principles_to_study=[top_principle],
            weekly_exercise=exercise,
            encouragement=(
                "Every game you play is a step forward — keep analysing and stay curious!"
            ),
        ))

    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# OpenRouter provider (official SDK, structured output via response_format)
# ---------------------------------------------------------------------------


class OpenRouterProvider(LLMProvider):
    """
    LLM provider backed by the official OpenRouter Python SDK.
    Uses response_format + Pydantic-derived JSON schema for structured output.
    API key: config value, falling back to OPENROUTER_API_KEY env var.
    """

    DEFAULT_MODEL = "anthropic/claude-haiku-4-5-20251001"

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._model = model or self.DEFAULT_MODEL
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, prompt: str, system: str = "") -> str:
        from openrouter import OpenRouter

        messages = _build_messages(system, prompt)
        with OpenRouter(api_key=self._api_key) as client:
            response = client.chat.send(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        content = response.choices[0].message.content
        return content if content is not None else ""

    def complete_structured(self, prompt: str, system: str, schema: type[T]) -> T:
        from openrouter import OpenRouter

        messages = _build_messages(system, prompt)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        }

        with OpenRouter(api_key=self._api_key) as client:
            response = client.chat.send(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                response_format=response_format,
            )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenRouter returned empty content for structured request")
        return schema.model_validate_json(content)

    def is_available(self) -> bool:
        return bool(self._api_key)


def _build_messages(system: str, user: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


# ---------------------------------------------------------------------------
# Review orchestration
# ---------------------------------------------------------------------------

_CLASSIFICATION_LABELS = {
    MoveClassification.BLUNDER: "blunder ??",
    MoveClassification.MISTAKE: "mistake ?",
    MoveClassification.INACCURACY: "inaccuracy ?!",
}

_THEME_KEYWORDS: dict[str, list[str]] = {
    "Hanging pieces": ["undefended", "free", "hanging", "take"],
    "Back rank weakness": ["back rank", "back-rank", "mate on 1"],
    "Tactics": ["tactic", "fork", "pin", "skewer", "discovered"],
    "Endgame technique": ["king", "pawn endgame", "endgame", "opposition"],
    "Opening principles": ["opening", "development", "center", "castle"],
}


class ReviewGenerator:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def generate(self, analysis: GameAnalysis, profile: PlayerProfile) -> GameReview:
        bad_moves = self._select_critical_moves(analysis)
        themes = self._extract_themes(analysis)

        system_prompt = self._render_system(profile)
        user_prompt = self._render_user(analysis, profile, bad_moves, themes)

        if isinstance(self._llm, StubLLMProvider):
            self._llm.set_context(
                self._build_stub_context(analysis, profile, bad_moves, themes)
            )

        llm_review: _LLMReview = self._llm.complete_structured(
            user_prompt, system_prompt, _LLMReview
        )

        return GameReview(
            game_id=analysis.game_id,
            opening_comment=llm_review.opening_comment,
            summary=llm_review.summary,
            critical_moments=self._enrich_moments(llm_review.critical_moments, analysis),
            principles_to_study=llm_review.principles_to_study,
            weekly_exercise=llm_review.weekly_exercise,
            encouragement=llm_review.encouragement,
        )

    # ------------------------------------------------------------------
    # Template rendering

    def _render_system(self, profile: PlayerProfile) -> str:
        tmpl = _jinja_env.get_template("system.j2")
        return tmpl.render(
            player_name=profile.player_name,
            level=profile.estimated_level,
            games_analyzed=profile.stats.games_analyzed,
        )

    def _render_user(
        self,
        analysis: GameAnalysis,
        profile: PlayerProfile,
        bad_moves: list[MoveAnalysis],
        themes: list[str],
    ) -> str:
        tmpl = _jinja_env.get_template("user.j2")

        critical_moves = [
            {
                "move_number": ma.move.move_number,
                "san": ma.move.san,
                "label": _CLASSIFICATION_LABELS.get(ma.classification, "error"),
                "best_san": ma.best_eval.best_san,
            }
            for ma in bad_moves
        ]

        # Top recurring principles from prior games (from blunder_themes in profile)
        prior_weak_principles = [
            _PRINCIPLE_FOR_THEME.get(t, t)
            for t in sorted(
                profile.stats.blunder_themes,
                key=lambda t: profile.stats.blunder_themes[t],
                reverse=True,
            )[:3]
        ]

        return tmpl.render(
            player_name=profile.player_name,
            level=profile.estimated_level,
            color=analysis.player_color.value,
            result=analysis.metadata.result.value,
            accuracy_pct=f"{analysis.accuracy_pct:.0f}",
            blunders=analysis.blunders,
            mistakes=analysis.mistakes,
            inaccuracies=analysis.inaccuracies,
            opening_name=analysis.metadata.opening_name or "",
            critical_moves=critical_moves,
            themes=themes,
            prior_weak_principles=prior_weak_principles,
        )

    # ------------------------------------------------------------------
    # Helpers

    def _select_critical_moves(self, analysis: GameAnalysis) -> list[MoveAnalysis]:
        bad = [
            ma for ma in analysis.move_analyses
            if ma.move.color == analysis.player_color
            and ma.classification in (MoveClassification.BLUNDER, MoveClassification.MISTAKE)
        ]
        bad.sort(key=lambda m: m.cp_loss, reverse=True)
        return bad[:3]

    def _extract_themes(self, analysis: GameAnalysis) -> list[str]:
        counts: dict[str, int] = {}
        bad = [
            ma for ma in analysis.move_analyses
            if ma.move.color == analysis.player_color
            and ma.classification in (MoveClassification.BLUNDER, MoveClassification.MISTAKE)
        ]
        for ma in bad:
            text = (ma.comment + " " + ma.best_eval.best_san).lower()
            for theme, keywords in _THEME_KEYWORDS.items():
                for kw in keywords:
                    if kw in text:
                        counts[theme] = counts.get(theme, 0) + 1
                        break

        if not counts:
            return ["Hanging pieces"] if (analysis.blunders + analysis.mistakes) > 2 else ["Tactics"]
        return sorted(counts, key=lambda t: counts[t], reverse=True)

    def _enrich_moments(
        self,
        llm_moments: list[_LLMMoment],
        analysis: GameAnalysis,
    ) -> list[CriticalMoment]:
        lookup: dict[tuple[int, str], MoveAnalysis] = {
            (ma.move.move_number, ma.move.san): ma
            for ma in analysis.move_analyses
        }
        enriched: list[CriticalMoment] = []
        for m in llm_moments:
            ma = lookup.get((m.move_number, m.san))
            enriched.append(
                CriticalMoment(
                    move_number=m.move_number,
                    san=m.san,
                    fen=ma.move.fen_before if ma else "",
                    classification=ma.classification if ma else MoveClassification.MISTAKE,
                    principle=m.principle,
                    what_happened=m.what_happened,
                    what_to_ask=m.what_to_ask,
                    best_move_san=m.best_move_san,
                )
            )
        return enriched

    def _build_stub_context(
        self,
        analysis: GameAnalysis,
        profile: PlayerProfile,
        bad_moves: list[MoveAnalysis],
        themes: list[str],
    ) -> dict[str, Any]:
        return {
            "level": profile.estimated_level,
            "player_name": profile.player_name,
            "blunders": analysis.blunders,
            "mistakes": analysis.mistakes,
            "inaccuracies": analysis.inaccuracies,
            "accuracy": analysis.accuracy_pct,
            "top_theme": themes[0] if themes else "Tactics",
            "opening_name": analysis.metadata.opening_name or "",
            "moments": [
                {
                    "move_number": ma.move.move_number,
                    "san": ma.move.san,
                    "comment": ma.comment or f"best was {ma.best_eval.best_san}",
                    "best_san": ma.best_eval.best_san,
                }
                for ma in bad_moves
            ],
        }
