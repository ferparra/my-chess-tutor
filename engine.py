from __future__ import annotations

import contextlib
import io
import platform
import re
import stat
import tarfile
import zipfile
from pathlib import Path
from typing import Literal, Optional

import chess
import chess.pgn
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from stockfish import Stockfish

from config import StockfishConfig
from models import (
    EngineEval,
    GameAnalysis,
    GameMetadata,
    Move,
    MoveAnalysis,
    MoveClassification,
    PlayerColor,
)

console = Console()

# Centipawn-loss thresholds (Lichess standard)
_CP_THRESHOLDS = [
    (MoveClassification.GOOD, 20),
    (MoveClassification.INACCURACY, 100),
    (MoveClassification.MISTAKE, 300),
]

# Accuracy formula: Lichess uses win-rate-based accuracy.
# We use a simpler approximation: max(0, 100 - cp_loss / 10) capped at 100.
def _cp_loss_to_accuracy(cp_loss: float) -> float:
    return max(0.0, min(100.0, 100.0 - cp_loss / 3.5))


class StockfishNotFoundError(Exception):
    pass


class StockfishEngine:
    def __init__(self, config: StockfishConfig) -> None:
        self._config = config
        binary = str(config.binary_path)
        if not Path(binary).exists():
            raise StockfishNotFoundError(
                f"Stockfish binary not found at {binary}. "
                "Run: chess-tutor setup stockfish"
            )
        self._sf = self._build_stockfish(binary)

    def _build_stockfish(self, binary: str) -> Stockfish:
        params = {
            "Threads": self._config.threads,
            "Hash": self._config.hash_mb,
            "MultiPV": self._config.num_top_moves,
        }
        return Stockfish(path=binary, depth=self._config.depth, parameters=params)

    def evaluate_position(self, fen: str) -> EngineEval:
        self._sf.set_fen_position(fen)
        top = self._sf.get_top_moves(self._config.num_top_moves)

        if not top:
            # No legal moves (checkmate/stalemate position)
            board = chess.Board(fen)
            return EngineEval(
                centipawns=0,
                depth=self._config.depth,
                best_uci="",
                best_san="",
                pv=[],
            )

        best = top[0]
        centipawns: Optional[int] = None
        mate_in: Optional[int] = None

        if best.get("Centipawn") is not None:
            centipawns = best["Centipawn"]
        if best.get("Mate") is not None:
            mate_in = best["Mate"]

        best_uci = best.get("Move", "")
        best_san = ""
        pv: list[str] = []

        # Convert best UCI to SAN
        if best_uci:
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(best_uci)
                best_san = board.san(move)
            except Exception:
                best_san = best_uci

        return EngineEval(
            centipawns=centipawns,
            mate_in=mate_in,
            depth=self._config.depth,
            best_uci=best_uci,
            best_san=best_san,
            pv=pv,
        )

    def classify_move(
        self,
        cp_loss: int,
        chess_move: chess.Move,
        board: chess.Board,
    ) -> MoveClassification:
        for classification, threshold in _CP_THRESHOLDS:
            if cp_loss <= threshold:
                return classification
        return MoveClassification.BLUNDER

    def analyze_game(
        self,
        pgn_text: str,
        player_color: PlayerColor,
        metadata: GameMetadata,
    ) -> GameAnalysis:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            raise ValueError("Could not parse PGN.")

        board = game.board()
        node = game
        move_analyses: list[MoveAnalysis] = []

        # Collect all moves first for progress bar
        moves_list = list(game.mainline_moves())
        total = len(moves_list)

        analysis = GameAnalysis(
            metadata=metadata,
            player_color=player_color,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing moves...", total=total)

            move_number = 0
            for chess_move in moves_list:
                color = PlayerColor.WHITE if board.turn == chess.WHITE else PlayerColor.BLACK
                move_number_int = board.fullmove_number
                fen_before = board.fen()

                # Evaluate position BEFORE the move (best move from this position)
                best_eval = self.evaluate_position(fen_before)

                # Play the move
                san = board.san(chess_move)
                uci = chess_move.uci()
                board.push(chess_move)

                # Evaluate position AFTER the move (from opponent's perspective, negate for us)
                fen_after = board.fen()
                after_eval = self.evaluate_position(fen_after)

                # Negate after_eval to get it from the perspective of the player who just moved
                played_cp = _negate_eval_cp(after_eval.centipawns)
                best_cp = best_eval.centipawns

                # Compute cp_loss (only meaningful for centipawn evals, not mate)
                cp_loss = 0
                if best_cp is not None and played_cp is not None:
                    cp_loss = max(0, best_cp - played_cp)
                elif best_eval.mate_in is not None and after_eval.mate_in is None:
                    # Missed a forced mate — treat as large blunder
                    cp_loss = 500
                elif best_eval.mate_in is None and after_eval.mate_in is not None:
                    # Walked into mate — treat as large blunder
                    cp_loss = 500

                # Build played_eval (re-frame after_eval from the mover's perspective)
                played_eval = EngineEval(
                    centipawns=played_cp,
                    mate_in=_negate_mate(after_eval.mate_in),
                    depth=after_eval.depth,
                    best_uci=after_eval.best_uci,
                    best_san=after_eval.best_san,
                    pv=after_eval.pv,
                )

                classification = self.classify_move(cp_loss, chess_move, board)

                move = Move(
                    uci=uci,
                    san=san,
                    move_number=move_number_int,
                    color=color,
                    fen_before=fen_before,
                )

                comment = _build_move_comment(san, cp_loss, classification, best_eval.best_san)

                move_analyses.append(
                    MoveAnalysis(
                        move=move,
                        played_eval=played_eval,
                        best_eval=best_eval,
                        cp_loss=cp_loss,
                        classification=classification,
                        comment=comment,
                    )
                )

                progress.advance(task)

        # Filter to only the player's moves for aggregate stats
        player_moves = [
            ma for ma in move_analyses if ma.move.color == player_color
        ]

        analysis.move_analyses = move_analyses
        analysis = _compute_aggregate_stats(analysis, player_moves)
        return analysis

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._sf.__del__()

    def __enter__(self) -> "StockfishEngine":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _negate_eval_cp(cp: Optional[int]) -> Optional[int]:
    if cp is None:
        return None
    return -cp


def _negate_mate(mate: Optional[int]) -> Optional[int]:
    if mate is None:
        return None
    return -mate


def _build_move_comment(
    san: str,
    cp_loss: int,
    classification: MoveClassification,
    best_san: str,
) -> str:
    if classification == MoveClassification.BRILLIANT:
        return f"{san} — excellent move!"
    if classification == MoveClassification.GOOD:
        return ""
    if classification == MoveClassification.INACCURACY:
        return f"{san} is slightly inaccurate ({cp_loss} cp). Consider {best_san}."
    if classification == MoveClassification.MISTAKE:
        return f"{san} is a mistake ({cp_loss} cp). {best_san} was better."
    return f"{san}?? — blunder ({cp_loss} cp). {best_san} was the right move."


def _compute_aggregate_stats(
    analysis: GameAnalysis,
    player_moves: list[MoveAnalysis],
) -> GameAnalysis:
    if not player_moves:
        return analysis

    analysis.total_moves = len(player_moves)
    analysis.blunders = sum(1 for m in player_moves if m.classification == MoveClassification.BLUNDER)
    analysis.mistakes = sum(1 for m in player_moves if m.classification == MoveClassification.MISTAKE)
    analysis.inaccuracies = sum(1 for m in player_moves if m.classification == MoveClassification.INACCURACY)
    analysis.average_cp_loss = sum(m.cp_loss for m in player_moves) / len(player_moves)
    analysis.accuracy_pct = _cp_loss_to_accuracy(analysis.average_cp_loss)
    return analysis


# ---------------------------------------------------------------------------
# Stockfish binary download
# ---------------------------------------------------------------------------

_GITHUB_LATEST = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"

_ASSET_PATTERNS: dict[str, list[str]] = {
    "darwin-arm64": ["macos-m1", "macos-arm", "mac-arm", "apple-silicon", "macos"],
    "darwin-x86_64": ["macos-x86", "macos-legacy", "macos"],
    "linux-x86_64": ["ubuntu", "linux", "x86-64"],
    "linux-aarch64": ["linux-arm", "aarch64", "arm64"],
    "windows-x86_64": ["windows", "win"],
}


def _detect_platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        return f"darwin-{machine}"
    if system == "linux":
        return f"linux-{machine}"
    if system == "windows":
        return "windows-x86_64"
    raise RuntimeError(f"Unsupported platform: {system}/{machine}")


def download_stockfish(dest_path: Path, verbose: bool = True) -> None:
    platform_key = _detect_platform_key()
    patterns = _ASSET_PATTERNS.get(platform_key, ["linux"])

    if verbose:
        console.print(f"[cyan]Fetching latest Stockfish release info...[/cyan]")

    resp = requests.get(_GITHUB_LATEST, timeout=15)
    resp.raise_for_status()
    release = resp.json()
    tag = release["tag_name"]
    assets = release.get("assets", [])

    if verbose:
        console.print(f"[cyan]Latest release: {tag}[/cyan]")

    # Find best matching asset
    chosen = None
    for pattern in patterns:
        for asset in assets:
            name = asset["name"].lower()
            if pattern in name and not name.endswith(".sha256") and not name.endswith(".md5"):
                chosen = asset
                break
        if chosen:
            break

    if chosen is None:
        names = [a["name"] for a in assets]
        raise RuntimeError(
            f"No Stockfish asset found for platform '{platform_key}'. "
            f"Available assets: {names}"
        )

    url = chosen["browser_download_url"]
    filename = chosen["name"]

    if verbose:
        console.print(f"[cyan]Downloading {filename}...[/cyan]")

    # Download with progress bar
    download_resp = requests.get(url, stream=True, timeout=60)
    download_resp.raise_for_status()
    total_size = int(download_resp.headers.get("content-length", 0))

    data = b""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Downloading {filename}", total=total_size or None)
        for chunk in download_resp.iter_content(chunk_size=8192):
            data += chunk
            progress.advance(task, len(chunk))

    # Extract binary
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    binary_name_re = re.compile(r"stockfish[^/\\]*", re.IGNORECASE)

    if filename.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            binary_entry = next(
                (n for n in zf.namelist() if binary_name_re.search(n) and not n.endswith("/")),
                None,
            )
            if binary_entry is None:
                raise RuntimeError(f"No Stockfish binary found inside {filename}")
            dest_path.write_bytes(zf.read(binary_entry))

    elif filename.endswith((".tar.gz", ".tgz", ".tar.bz2")):
        tar_mode: Literal["r:gz", "r:bz2"] = (
            "r:gz" if filename.endswith((".tar.gz", ".tgz")) else "r:bz2"
        )
        with tarfile.open(fileobj=io.BytesIO(data), mode=tar_mode) as tf:
            tar_entry: tarfile.TarInfo | None = next(
                (m for m in tf.getmembers() if binary_name_re.search(m.name) and m.isfile()),
                None,
            )
            if tar_entry is None:
                raise RuntimeError(f"No Stockfish binary found inside {filename}")
            f = tf.extractfile(tar_entry)
            if f is None:
                raise RuntimeError("Could not extract binary")
            dest_path.write_bytes(f.read())
    else:
        # Plain binary
        dest_path.write_bytes(data)

    # Make executable
    dest_path.chmod(dest_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    if verbose:
        console.print(f"[green]Stockfish installed to {dest_path}[/green]")
