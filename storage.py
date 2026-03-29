from __future__ import annotations

from datetime import datetime
from pathlib import Path

from config import AppConfig
from models import GameRecord, PlayerProfile, PlayerStats


class StorageError(Exception):
    pass


class GameStore:
    def __init__(self, config: AppConfig) -> None:
        self._games_dir = config.data_dir / "games"
        self._games_dir.mkdir(parents=True, exist_ok=True)

    def save_game(self, record: GameRecord) -> Path:
        path = self._games_dir / f"{record.analysis.game_id}.json"
        path.write_text(record.model_dump_json(indent=2))
        return path

    def load_game(self, game_id: str) -> GameRecord:
        path = self._games_dir / f"{game_id}.json"
        if not path.exists():
            raise StorageError(f"Game not found: {game_id}")
        try:
            return GameRecord.model_validate_json(path.read_text())
        except Exception as e:
            raise StorageError(f"Corrupt game file {game_id}: {e}") from e

    def list_games(self) -> list[str]:
        files = sorted(self._games_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem for p in files]

    def delete_game(self, game_id: str) -> None:
        path = self._games_dir / f"{game_id}.json"
        if not path.exists():
            raise StorageError(f"Game not found: {game_id}")
        path.unlink()


class ProfileStore:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._path = config.data_dir / "player_profile.json"
        config.data_dir.mkdir(parents=True, exist_ok=True)

    def load_profile(self) -> PlayerProfile:
        if not self._path.exists():
            profile = PlayerProfile(player_name=self._config.player_name)
            self.save_profile(profile)
            return profile
        try:
            return PlayerProfile.model_validate_json(self._path.read_text())
        except Exception as e:
            raise StorageError(f"Corrupt profile file: {e}") from e

    def save_profile(self, profile: PlayerProfile) -> None:
        profile.last_updated = datetime.utcnow()
        self._path.write_text(profile.model_dump_json(indent=2))

    def update_stats_from_analysis(
        self,
        profile: PlayerProfile,
        record: GameRecord,
    ) -> PlayerProfile:
        analysis = record.analysis
        s = profile.stats
        n = s.games_analyzed + 1

        s.games_analyzed = n
        s.total_blunders += analysis.blunders
        s.total_mistakes += analysis.mistakes
        s.total_inaccuracies += analysis.inaccuracies

        # Rolling means
        s.average_cp_loss_all = (s.average_cp_loss_all * (n - 1) + analysis.average_cp_loss) / n
        s.average_accuracy_all = (s.average_accuracy_all * (n - 1) + analysis.accuracy_pct) / n

        # Theme frequency from review — use the principles the LLM flagged
        if record.review:
            for principle in record.review.principles_to_study:
                s.blunder_themes[principle] = s.blunder_themes.get(principle, 0) + 1

        profile.estimated_level = self._classify_level(s)

        if analysis.game_id not in profile.game_ids:
            profile.game_ids.append(analysis.game_id)

        return profile

    def _classify_level(self, stats: PlayerStats) -> str:
        acc = stats.average_accuracy_all
        if acc >= 80.0:
            return "advanced"
        elif acc >= 65.0:
            return "intermediate"
        return "beginner"
