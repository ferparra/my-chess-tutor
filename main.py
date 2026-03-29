from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from config import LLMConfig, load_config, save_config
from engine import StockfishEngine, StockfishNotFoundError, download_stockfish
from board_renderer import embed_review_images
from fetcher import ChessComError, ChessComFetcher, ChessComGame
from models import (
    GameAnalysis,
    GameMetadata,
    GameRecord,
    GameResult,
    GameReview,
    MoveClassification,
    PlayerColor,
    PlayerProfile,
    PlayerStats,
)
from reviewer import LLMProvider, OpenRouterProvider, ReviewGenerator, StubLLMProvider

_VERSION = "0.1.0"

_HELP = """\
A personal chess tutor: Stockfish analysis + human-friendly reviews.

[bold]Quick start:[/bold]

  [cyan]chess-tutor setup stockfish[/cyan]               # download Stockfish binary
  [cyan]chess-tutor setup config --player-name You[/cyan] # set your name
  [cyan]chess-tutor analyze game.pgn --color white[/cyan] # analyse a local PGN
  [cyan]chess-tutor fetch ferparra --analyze -n 3[/cyan]  # fetch & analyse from chess.com

JSON output goes to [bold]stdout[/bold]; progress and display go to [bold]stderr[/bold].
Pipe [cyan]chess-tutor analyze[/cyan] or [cyan]fetch --analyze[/cyan] into [cyan]jq[/cyan] or any JSON consumer.
"""


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"chess-tutor {_VERSION}")
        raise typer.Exit()


app = typer.Typer(
    name="chess-tutor",
    help=_HELP,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
)
setup_app = typer.Typer(
    help="Installation and configuration commands.",
    rich_markup_mode="rich",
)
app.add_typer(setup_app, name="setup")


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    pass


# All Rich display goes to stderr so stdout stays clean for JSON output.
console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Classification display helpers
# ---------------------------------------------------------------------------

_CLASS_STYLE: dict[MoveClassification, tuple[str, str]] = {
    MoveClassification.BRILLIANT: ("★", "bright_cyan"),
    MoveClassification.GOOD: ("✓", "green"),
    MoveClassification.INACCURACY: ("?!", "yellow"),
    MoveClassification.MISTAKE: ("?", "orange3"),
    MoveClassification.BLUNDER: ("??", "red"),
}


# ---------------------------------------------------------------------------
# setup commands
# ---------------------------------------------------------------------------


@setup_app.command("stockfish")
def setup_stockfish(
    path: Optional[str] = typer.Option(
        None, "--path",
        help=(
            "Path to an existing Stockfish binary. "
            "If omitted the latest release is downloaded from GitHub."
        ),
    ),
) -> None:
    """Download the latest Stockfish binary, or register one you already have."""
    config = load_config()

    if path:
        binary = Path(path).expanduser().resolve()
        if not binary.exists():
            console.print(f"[red]Binary not found: {binary}[/red]")
            raise typer.Exit(1)
        config.stockfish.binary_path = binary
        save_config(config)
        console.print(f"[green]Stockfish configured at {binary}[/green]")
        return

    dest = config.stockfish.binary_path
    try:
        download_stockfish(dest)
        config.stockfish.binary_path = dest
        save_config(config)
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1)


@setup_app.command("config")
def setup_config(
    player_name: Optional[str] = typer.Option(
        None, "--player-name",
        help="Your display name shown in reviews.",
    ),
    username: Optional[str] = typer.Option(
        None, "--username",
        help="Your chess.com username (used by [cyan]fetch[/cyan]).",
    ),
    depth: Optional[int] = typer.Option(
        None, "--depth", min=1, max=30,
        help="Default Stockfish search depth (1-30). Higher = stronger but slower.",
    ),
    llm_provider: Optional[str] = typer.Option(
        None, "--llm-provider",
        help="LLM backend: [bold]stub[/bold] (offline template) or [bold]openrouter[/bold].",
    ),
    llm_model: Optional[str] = typer.Option(
        None, "--llm-model",
        help="OpenRouter model ID, e.g. [dim]anthropic/claude-haiku-4-5-20251001[/dim].",
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        help="OpenRouter API key. Also read from [bold]OPENROUTER_API_KEY[/bold] env var.",
    ),
) -> None:
    """Update app configuration. Only provided options are changed."""
    config = load_config()

    if player_name:
        config.player_name = player_name
    if username:
        config.chesscom_username = username
    if depth is not None:
        config.stockfish.depth = depth
    if llm_provider:
        config.llm.provider = llm_provider
    if llm_model:
        config.llm.model = llm_model
    if api_key:
        config.llm.api_key = api_key

    save_config(config)
    console.print(Panel(config.model_dump_json(indent=2), title="[bold]Current Config[/bold]"))


# ---------------------------------------------------------------------------
# LLM provider factory
# ---------------------------------------------------------------------------


def _build_llm_provider(llm_config: LLMConfig) -> LLMProvider:
    if llm_config.provider == "openrouter":
        provider = OpenRouterProvider(
            api_key=llm_config.api_key,
            model=llm_config.model,
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
        )
        if not provider.is_available():
            console.print(
                "[yellow]No OpenRouter API key found. Set --api-key or "
                "OPENROUTER_API_KEY env var. Falling back to stub.[/yellow]"
            )
            return StubLLMProvider()
        return provider
    return StubLLMProvider()


def _default_profile(config_player_name: str) -> PlayerProfile:
    return PlayerProfile(
        player_name=config_player_name,
        stats=PlayerStats(),
    )


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@app.command()
def analyze(
    pgn: str = typer.Argument(..., help="Path to a PGN file."),
    color: PlayerColor = typer.Option(
        PlayerColor.WHITE, "--color",
        help="Which side you played.",
        show_default=True,
    ),
    depth: Optional[int] = typer.Option(
        None, "--depth", min=1, max=30,
        help="Stockfish search depth (1-30). Overrides the configured default.",
        show_default=False,
    ),
    no_review: bool = typer.Option(
        False, "--no-review",
        help="Skip the LLM review; output engine stats only.",
    ),
) -> None:
    """Analyze a game from a PGN file. Outputs GameRecord JSON to stdout."""
    pgn_path = Path(pgn).expanduser()
    if not pgn_path.exists():
        console.print(f"[red]PGN file not found: {pgn_path}[/red]")
        raise typer.Exit(1)

    player_color = color

    config = load_config()
    if depth is not None:
        config.stockfish.depth = depth

    pgn_text = pgn_path.read_text()
    metadata = _extract_metadata(pgn_text, pgn_path)

    console.print(
        Panel(
            f"[bold]{metadata.white_player}[/bold] vs [bold]{metadata.black_player}[/bold]"
            f"  |  {metadata.date or 'unknown date'}"
            f"  |  Analyzing as [cyan]{player_color.value}[/cyan]",
            title="Game",
        )
    )

    try:
        with StockfishEngine(config.stockfish) as engine:
            game_analysis = engine.analyze_game(pgn_text, player_color, metadata)
    except StockfishNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    profile = _default_profile(config.player_name)

    review = None
    if not no_review:
        llm = _build_llm_provider(config.llm)
        generator = ReviewGenerator(llm)
        review = generator.generate(game_analysis, profile)
        embed_review_images(review)

    record = GameRecord(analysis=game_analysis, review=review, pgn=pgn_text)

    _print_move_table(game_analysis, player_color)
    if review:
        _print_review(review)

    sys.stdout.write(record.model_dump_json() + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

_TIME_CLASSES = ("bullet", "blitz", "rapid", "classical")


@app.command()
def fetch(
    username: Optional[str] = typer.Argument(
        None,
        help="chess.com username. Falls back to [cyan]--username[/cyan] set in [cyan]setup config[/cyan].",
    ),
    limit: int = typer.Option(
        10, "--limit", "-n",
        help="Maximum number of games to fetch.",
        show_default=True,
    ),
    time_class: Optional[str] = typer.Option(
        None, "--time-class", "-t",
        help="Filter by time control: bullet, blitz, rapid, or classical.",
    ),
    month: Optional[str] = typer.Option(
        None, "--month", "-m",
        help="Fetch games from a specific month, e.g. [bold]2026/03[/bold]. Defaults to the most recent month.",
    ),
    analyze_games: bool = typer.Option(
        False, "--analyze", "-a",
        help="Run Stockfish + LLM review on each game. Outputs NDJSON GameRecords to stdout.",
    ),
    no_review: bool = typer.Option(
        False, "--no-review",
        help="Skip the LLM review when [cyan]--analyze[/cyan] is set.",
    ),
) -> None:
    """Fetch recent games from chess.com. With --analyze, outputs NDJSON GameRecords to stdout."""
    config = load_config()
    resolved_username = username or config.chesscom_username
    if not resolved_username:
        console.print(
            "[red]No chess.com username provided. "
            "Pass it as an argument or run: chess-tutor setup config --username <name>[/red]"
        )
        raise typer.Exit(1)

    if time_class and time_class not in _TIME_CLASSES:
        console.print(f"[red]Invalid --time-class '{time_class}'. Choose from: {', '.join(_TIME_CLASSES)}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Fetching games for [bold]{resolved_username}[/bold] from chess.com…[/cyan]")

    try:
        with ChessComFetcher() as fetcher:
            games = fetcher.get_recent_games(
                resolved_username,
                limit=limit,
                time_class=time_class,
                month=month,
            )
    except ChessComError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if not games:
        console.print("[yellow]No games found for the given filters.[/yellow]")
        return

    _print_fetched_games(games, resolved_username)

    if not analyze_games:
        console.print("\n[dim]Tip: add --analyze to run Stockfish + review on each game.[/dim]")
        return

    profile = _default_profile(config.player_name)

    try:
        engine_ctx = StockfishEngine(config.stockfish)
    except StockfishNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    llm = _build_llm_provider(config.llm)
    generator = ReviewGenerator(llm)

    with engine_ctx as engine:
        for i, game in enumerate(games, 1):
            color = game.player_color(resolved_username)
            opponent = game.opponent(resolved_username)
            date_str = game.pgn_date() or "?"
            console.rule(
                f"[bold]Game {i}/{len(games)}: {resolved_username} ({color.value})"
                f" vs {opponent.username} — {date_str}[/bold]"
            )

            metadata = _extract_metadata_from_str(game.pgn)
            metadata.opening_name = game.opening_name()

            console.print(Panel(
                f"[bold]{metadata.white_player}[/bold] vs [bold]{metadata.black_player}[/bold]"
                f"  |  {date_str}  |  {game.time_class}  |  Analyzing as [cyan]{color.value}[/cyan]",
                title="Game",
            ))

            try:
                game_analysis = engine.analyze_game(game.pgn, color, metadata)
            except Exception as e:
                console.print(f"[red]Analysis failed: {e}[/red]")
                continue

            review = None
            if not no_review:
                try:
                    review = generator.generate(game_analysis, profile)
                    embed_review_images(review)
                except Exception as e:
                    console.print(f"[yellow]Review failed: {e}[/yellow]")

            record = GameRecord(analysis=game_analysis, review=review, pgn=game.pgn)

            _print_move_table(game_analysis, color)
            if review:
                _print_review(review)

            sys.stdout.write(record.model_dump_json() + "\n")
            sys.stdout.flush()

    console.print(f"\n[green]Done. {len(games)} game(s) fetched and analyzed.[/green]")


def _print_fetched_games(games: list[ChessComGame], username: str) -> None:
    table = Table(
        title=f"Recent games for [bold]{username}[/bold]",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("Date", style="dim")
    table.add_column("Type")
    table.add_column("Opponent")
    table.add_column("Color")
    table.add_column("Result")
    table.add_column("Rated")

    for g in games:
        color = g.player_color(username)
        opponent = g.opponent(username)
        result = g.result_for(username)
        result_style = {"win": "green", "loss": "red", "draw": "yellow"}.get(result, "white")
        table.add_row(
            g.pgn_date() or g.end_datetime().strftime("%Y.%m.%d"),
            g.time_class,
            f"{opponent.username} ({opponent.rating})",
            color.value,
            f"[{result_style}]{result}[/{result_style}]",
            "✓" if g.rated else "–",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Internal display helpers
# ---------------------------------------------------------------------------


def _print_move_table(analysis: GameAnalysis, player_color: PlayerColor) -> None:
    table = Table(
        title=f"Move Analysis — {player_color.value.capitalize()}",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
    )
    table.add_column("#", style="dim", justify="right")
    table.add_column("Move")
    table.add_column("Quality")
    table.add_column("CP loss", justify="right")
    table.add_column("Best", style="dim")
    table.add_column("Note")

    player_moves = [
        ma for ma in analysis.move_analyses if ma.move.color == player_color
    ]

    for ma in player_moves:
        symbol, style = _CLASS_STYLE.get(ma.classification, ("", "white"))
        best = ma.best_eval.best_san if ma.classification != MoveClassification.GOOD else ""
        table.add_row(
            str(ma.move.move_number),
            f"[bold]{ma.move.san}[/bold]",
            f"[{style}]{symbol} {ma.classification.value}[/{style}]",
            str(ma.cp_loss) if ma.cp_loss else "",
            best,
            ma.comment[:60] if ma.comment else "",
        )

    console.print(table)

    console.print(
        f"  Accuracy: [bold]{analysis.accuracy_pct:.0f}%[/bold]  |  "
        f"Blunders: [red]{analysis.blunders}[/red]  |  "
        f"Mistakes: [orange3]{analysis.mistakes}[/orange3]  |  "
        f"Inaccuracies: [yellow]{analysis.inaccuracies}[/yellow]\n"
    )


def _print_review(review: GameReview) -> None:
    console.print(Panel(review.opening_comment, title="[bold]Opening Phase[/bold]"))
    console.print(Panel(review.summary, title="[bold]Game Summary[/bold]"))

    if review.critical_moments:
        for m in review.critical_moments:
            symbol, style = _CLASS_STYLE.get(m.classification, ("", "white"))
            header = (
                f"[{style}]{symbol} Move {m.move_number}: {m.san}[/{style}]"
                f"  —  Principle: [bold]{m.principle}[/bold]"
            )
            body = (
                f"{m.what_happened}\n\n"
                f"[dim]Better:[/dim] [green]{m.best_move_san}[/green]\n\n"
                f"[italic]Ask yourself: {m.what_to_ask}[/italic]"
            )
            console.print(Panel(body, title=header))

    if review.principles_to_study:
        console.print(
            f"[bold]Principles to study:[/bold] {', '.join(review.principles_to_study)}\n"
        )

    console.print(
        Panel(
            f"[bold]This week's exercise:[/bold]\n{review.weekly_exercise}\n\n"
            f"[italic]{review.encouragement}[/italic]",
            title="[cyan]Next Steps[/cyan]",
        )
    )


# ---------------------------------------------------------------------------
# Metadata extraction helpers
# ---------------------------------------------------------------------------


def _extract_metadata(pgn_text: str, pgn_path: Path) -> GameMetadata:
    return _extract_metadata_from_str(pgn_text, source=str(pgn_path))


def _extract_metadata_from_str(pgn_text: str, source: str = "") -> GameMetadata:
    import chess.pgn
    import io

    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return GameMetadata(pgn_source=source)

    headers = game.headers
    result_str = headers.get("Result", "*")
    try:
        result = GameResult(result_str)
    except ValueError:
        result = GameResult.UNKNOWN

    return GameMetadata(
        white_player=headers.get("White", "?"),
        black_player=headers.get("Black", "?"),
        date=headers.get("Date"),
        event=headers.get("Event"),
        result=result,
        opening_name=headers.get("Opening"),
        pgn_source=source,
    )


if __name__ == "__main__":
    app()
