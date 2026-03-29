from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import requests
from pydantic import BaseModel, Field, ValidationError, model_validator

from models import PlayerColor

_BASE = "https://api.chess.com/pub/player"
_HEADERS = {
    "User-Agent": "my-chess-tutor/0.1.0 (https://github.com/ferparra/my-chess-tutor)",
    "Accept": "application/json",
}

# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------


class ChessComPlayer(BaseModel):
    username: str
    rating: int
    result: str
    uuid: str = ""


class ChessComGame(BaseModel):
    url: str
    pgn: str
    time_class: str
    time_control: str
    rated: bool
    white: ChessComPlayer
    black: ChessComPlayer
    uuid: str = ""
    end_time: int = 0
    eco: Optional[str] = None
    accuracies: Optional[dict[str, float]] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: dict) -> dict:
        # ECOUrl comes from the API as a field but we don't store it
        data.pop("ECOUrl", None)
        data.pop("tcn", None)
        data.pop("initial_setup", None)
        data.pop("fen", None)
        data.pop("rules", None)
        return data

    def player_color(self, username: str) -> PlayerColor:
        if self.white.username.lower() == username.lower():
            return PlayerColor.WHITE
        return PlayerColor.BLACK

    def result_for(self, username: str) -> str:
        """Returns 'win', 'loss', or 'draw' from the player's perspective."""
        color = self.player_color(username)
        side = self.white if color == PlayerColor.WHITE else self.black
        r = side.result
        if r == "win":
            return "win"
        if r in ("agreed", "repetition", "stalemate", "insufficient", "50move", "timevsinsufficient"):
            return "draw"
        return "loss"

    def end_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.end_time, tz=timezone.utc)

    def opponent(self, username: str) -> ChessComPlayer:
        color = self.player_color(username)
        return self.black if color == PlayerColor.WHITE else self.white

    def opening_name(self) -> Optional[str]:
        """Extract opening name from ECO URL if available via PGN headers."""
        for line in self.pgn.splitlines():
            if line.startswith('[Opening "'):
                return line[10:].rstrip('"]')
        return None

    def pgn_date(self) -> Optional[str]:
        for line in self.pgn.splitlines():
            if line.startswith('[Date "'):
                return line[7:].rstrip('"]')
        return None


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


class ChessComError(Exception):
    pass


class ChessComFetcher:
    def __init__(self, timeout: int = 15) -> None:
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._timeout = timeout

    def get_archives(self, username: str) -> list[str]:
        """Return all monthly archive URLs for a player, newest last."""
        url = f"{_BASE}/{username.lower()}/games/archives"
        resp = self._get(url)
        return resp.get("archives", [])

    def get_monthly_games(self, archive_url: str) -> list[ChessComGame]:
        """Fetch all games from a monthly archive URL."""
        data = self._get(archive_url)
        games: list[ChessComGame] = []
        for raw in data.get("games", []):
            try:
                games.append(ChessComGame.model_validate(raw))
            except ValidationError:
                continue  # skip malformed game entries from the API
        return games

    def get_recent_games(
        self,
        username: str,
        limit: int = 10,
        time_class: Optional[str] = None,
        month: Optional[str] = None,
    ) -> list[ChessComGame]:
        """
        Return up to `limit` recent games for a player.

        Args:
            username: chess.com username
            limit: max games to return
            time_class: filter by 'bullet', 'blitz', 'rapid', 'classical' (None = all)
            month: specific month as 'YYYY/MM'; if None, walks archives newest-first
        """
        archives = self.get_archives(username)
        if not archives:
            raise ChessComError(f"No game archives found for '{username}'")

        if month:
            # Find the matching archive URL
            suffix = month.replace("-", "/")
            target = next((a for a in archives if a.endswith(suffix)), None)
            if target is None:
                raise ChessComError(f"No archive found for month '{month}'")
            candidates = [target]
        else:
            # Walk newest-first
            candidates = list(reversed(archives))

        collected: list[ChessComGame] = []
        for archive_url in candidates:
            games = self.get_monthly_games(archive_url)
            # Newest first within the month
            games.sort(key=lambda g: g.end_time, reverse=True)
            for g in games:
                if time_class and g.time_class != time_class:
                    continue
                collected.append(g)
                if len(collected) >= limit:
                    return collected

        return collected

    def _get(self, url: str) -> dict:
        try:
            resp = self._session.get(url, timeout=self._timeout)
        except requests.RequestException as e:
            raise ChessComError(f"Request failed: {e}") from e
        if resp.status_code == 404:
            raise ChessComError(f"Not found: {url}")
        if not resp.ok:
            raise ChessComError(f"HTTP {resp.status_code} for {url}")
        return resp.json()

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "ChessComFetcher":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
