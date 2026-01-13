"""CLI entry point for audiobook-optimizer."""

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from audiobook_optimizer.config import ai_available, get_settings
from audiobook_optimizer.domain.models import ProcessingStatus
from audiobook_optimizer.services.processor import AudiobookProcessor

app = typer.Typer(
    name="audiobook-optimizer",
    help="Scan, convert, and organize audiobooks to M4B format.",
    no_args_is_help=True,
)
console = Console()


# Global state for flags
class State:
    ai_enabled: bool = False
    debug: bool = False


state = State()


def print_cache_debug() -> None:
    """Print cachekit cache metrics for debugging."""
    if not state.debug:
        return

    console.print("\n[dim]─── Cache Debug ───[/dim]")

    # Get stats from cached functions via cache_info()
    from audiobook_optimizer.adapters.ffmpeg import _probe_file_cached

    stats: list[tuple[str, int, int, int, int]] = []  # (name, l1, l2, hits, misses)

    # ffprobe cache
    try:
        info = _probe_file_cached.cache_info()
        stats.append(("ffprobe", info.l1_hits, info.l2_hits, info.hits, info.misses))
    except Exception:
        pass

    # AI verification cache (only if used)
    try:
        from audiobook_optimizer.adapters.ai_batch import _verify_batch_cached

        info = _verify_batch_cached.cache_info()
        if info.hits + info.misses > 0:
            stats.append(("ai_verify", info.l1_hits, info.l2_hits, info.hits, info.misses))
    except Exception:
        pass

    # Display stats
    for name, l1, l2, hits, misses in stats:
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0
        location = f"L1:{l1} L2:{l2}" if l2 > 0 else f"L1:{l1}"
        console.print(f"  [cyan]{name}[/cyan]: {hits} hits ({location}), {misses} misses ([green]{hit_rate:.0f}%[/green])")

    if not stats:
        console.print("  [dim]No cache operations recorded[/dim]")

    # Show Redis connection status
    redis_url = os.getenv("CACHEKIT_REDIS_URL") or os.getenv("REDIS_URL")
    if redis_url:
        # Mask password if present
        display_url = redis_url.split("@")[-1] if "@" in redis_url else redis_url
        console.print(f"  [dim]Redis: {display_url}[/dim]")
    else:
        console.print("  [dim]Redis: not configured (L1 only)[/dim]")

    console.print("[dim]───────────────────[/dim]")


@app.callback()
def main(
    ai: bool | None = typer.Option(
        None,
        "--ai/--no-ai",
        help="Enable/disable AI verification. Default: auto (on if ANTHROPIC_API_KEY set)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-vv",
        help="Show cache debug info (hits/misses) at end of operation",
    ),
):
    """Global options for audiobook-optimizer."""
    state.debug = debug

    if ai is None:
        # Auto-detect from settings (validator ensures ai_enabled is never None)
        settings = get_settings()
        state.ai_enabled = settings.ai_enabled or False
    else:
        state.ai_enabled = ai

    if state.ai_enabled and not ai_available():
        console.print("[yellow]Warning: --ai requested but ANTHROPIC_API_KEY not set. Disabling AI.[/yellow]")
        state.ai_enabled = False


def format_duration(ms: int) -> str:
    """Format milliseconds as HH:MM:SS."""
    seconds = ms // 1000
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_size(bytes_: int | float) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} TB"


def format_bitrate(kbps: int | None) -> str:
    """Format bitrate."""
    if kbps is None:
        return "unknown"
    return f"{kbps} kbps"


@app.command()
def scan(
    source: Path = typer.Argument(
        ...,
        help="Directory to scan for audiobooks",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed file info"),
) -> None:
    """Scan directory for audiobooks without processing."""
    processor = AudiobookProcessor(
        source_dir=source,
        output_dir=Path("/tmp"),  # Won't be used
    )

    if verbose:
        console.print("Scanning directories...")
        sources = []
        for src in processor.scanner.scan_directory(source):
            console.print(f"  Found: [cyan]{src.source_path.name}[/cyan] ({src.file_count} files)")
            sources.append(src)
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Scanning for audiobooks...", total=None)
            sources = processor.scan()

    if not sources:
        console.print("[yellow]No audiobooks found.[/yellow]")
        return

    if verbose:
        # Detailed view with file info
        from audiobook_optimizer.adapters.ffmpeg import FFmpegConverter

        converter = FFmpegConverter()
        target_bitrate = 64  # Default target

        # First pass: collect all metadata and quality info
        scan_results = []
        console.print(f"\nAnalyzing {len(sources)} audiobook(s)...")
        for i, src in enumerate(sources, 1):
            console.print(f"  [{i}/{len(sources)}] Analyzing: [cyan]{src.source_path.name}[/cyan]")
            processor._populate_file_info(src)
            src.metadata = processor.metadata_extractor.infer_metadata(src)

            # Calculate bitrate stats
            bitrates = [f.bitrate for f in src.audio_files if f.bitrate]
            min_bitrate = min(bitrates) if bitrates else None
            avg_bitrate = sum(bitrates) // len(bitrates) if bitrates else None

            # Determine what would happen on processing
            can_remux, _ = converter._can_remux(src.audio_files, mono=True)
            if can_remux:
                action = "remux"
                effective_bitrate = avg_bitrate or 0
            else:
                effective_bitrate = converter._calculate_effective_bitrate(src.audio_files, target_bitrate)
                action = "transcode"

            quality_info = {
                "bitrate": avg_bitrate,
                "min_bitrate": min_bitrate,
                "effective_bitrate": effective_bitrate,
                "action": action,
            }
            scan_results.append((src, src.metadata, quality_info))

        # AI batch verification if enabled
        ai_corrections = {}
        if state.ai_enabled:
            try:
                from audiobook_optimizer.adapters.ai_batch import BatchAIVerifier

                console.print(f"\nAI verifying {len(scan_results)} audiobooks...")
                verifier = BatchAIVerifier()
                ai_corrections = verifier.verify_batch(scan_results)
                if ai_corrections:
                    console.print(f"[dim]AI suggested corrections for {len(ai_corrections)} audiobook(s)[/dim]\n")
            except Exception as e:
                console.print(f"[yellow]AI verification failed: {e}[/yellow]\n")

        # Display results
        for i, (src, metadata, quality_info) in enumerate(scan_results):
            min_bitrate = quality_info["min_bitrate"]
            avg_bitrate = quality_info["bitrate"]
            effective_bitrate = quality_info["effective_bitrate"]
            action = quality_info["action"]

            # Format action string
            if action == "remux":
                action_str = "[cyan]remux[/cyan] (no re-encoding)"
            elif min_bitrate and effective_bitrate < target_bitrate:
                action_str = f"[yellow]transcode[/yellow] → {effective_bitrate}kbps (capped by source)"
            else:
                action_str = f"[dim]transcode[/dim] → {effective_bitrate}kbps"

            console.print(f"\n[bold cyan]{src.source_path.name}[/bold cyan]")
            console.print(f"  Files:    {src.file_count} × {src.primary_format.value.upper()}")
            console.print(f"  Size:     {format_size(src.total_size_bytes)}")
            console.print(f"  Duration: {format_duration(src.total_duration_ms)}")
            console.print(f"  Bitrate:  {format_bitrate(avg_bitrate)} (min: {format_bitrate(min_bitrate)})")
            console.print(f"  Output:   {action_str}")

            # Show metadata with AI corrections if any
            if i in ai_corrections:
                ai = ai_corrections[i]
                console.print(f"  → Title:  {ai.title}")
                console.print(f"  → Author: {ai.author}")
                if ai.series:
                    console.print(f"  → Series: {ai.series} #{ai.series_number}")
                if ai.changes:
                    console.print(f"  [yellow]AI corrections: {', '.join(ai.changes)}[/yellow]")
                if ai.quality_note:
                    console.print(f"  [yellow]⚠ {ai.quality_note}[/yellow]")
            elif metadata:
                console.print(f"  → Title:  {metadata.title}")
                console.print(f"  → Author: {metadata.author}")
                if metadata.series:
                    console.print(f"  → Series: {metadata.series} #{metadata.series_number}")
    else:
        # Table view
        table = Table(title=f"Found {len(sources)} Audiobook(s)")
        table.add_column("Folder", style="cyan")
        table.add_column("Files", justify="right")
        table.add_column("Format")

        for src in sources:
            table.add_row(
                src.source_path.name,
                str(src.file_count),
                src.primary_format.value.upper(),
            )

        console.print(table)

    print_cache_debug()


@app.command()
def process(
    source: Path = typer.Argument(
        ...,
        help="Directory to scan for audiobooks",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
    output: Path = typer.Argument(
        ...,
        help="Library directory for processed audiobooks",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
    bitrate: int = typer.Option(64, "--bitrate", "-b", help="Target bitrate in kbps"),
    stereo: bool = typer.Option(False, "--stereo", help="Keep stereo (default: mono)"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be done"),
) -> None:
    """Process audiobooks: convert to M4B and organize."""
    # Show AI status
    if state.ai_enabled:
        console.print("[dim]AI verification enabled[/dim]")

    processor = AudiobookProcessor(
        source_dir=source,
        output_dir=output,
        bitrate=bitrate,
        mono=not stereo,
    )

    # Scan
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Scanning for audiobooks...", total=None)
        sources = processor.scan()

    if not sources:
        console.print("[yellow]No audiobooks found.[/yellow]")
        return

    console.print(f"Found [cyan]{len(sources)}[/cyan] audiobook(s)\n")

    # Populate file info and infer metadata for all sources
    for src in sources:
        processor._populate_file_info(src)
        src.metadata = processor.metadata_extractor.infer_metadata(src)

    # Batch AI verification if enabled
    ai_corrections = {}
    if state.ai_enabled:
        try:
            from audiobook_optimizer.adapters.ai_batch import BatchAIVerifier, apply_verification
            from audiobook_optimizer.adapters.ffmpeg import FFmpegConverter

            converter = FFmpegConverter()
            batch_data = []
            for src in sources:
                bitrates = [f.bitrate for f in src.audio_files if f.bitrate]
                avg_bitrate = sum(bitrates) // len(bitrates) if bitrates else 0
                can_remux, _ = converter._can_remux(src.audio_files, mono=not stereo)
                effective_bitrate = (
                    avg_bitrate if can_remux else converter._calculate_effective_bitrate(src.audio_files, bitrate)
                )
                quality_info = {
                    "bitrate": avg_bitrate,
                    "effective_bitrate": effective_bitrate,
                    "action": "remux" if can_remux else "transcode",
                }
                batch_data.append((src, src.metadata, quality_info))

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(f"AI verifying {len(sources)} audiobooks...", total=None)
                verifier = BatchAIVerifier()
                ai_corrections = verifier.verify_batch(batch_data)

            # Apply corrections
            for i, src in enumerate(sources):
                if i in ai_corrections and src.metadata is not None:
                    src.metadata = apply_verification(src.metadata, ai_corrections[i])

            if ai_corrections:
                console.print(f"[dim]AI corrected {len(ai_corrections)} audiobook(s)[/dim]\n")
        except Exception as e:
            console.print(f"[yellow]AI verification failed: {e}[/yellow]\n")

    if dry_run:
        _show_dry_run(processor, sources, output)
        return

    # Process each
    results = []
    total_source_bytes = 0
    total_output_bytes = 0

    for i, src in enumerate(sources, 1):
        console.print(f"[bold][{i}/{len(sources)}] Processing:[/bold] {src.source_path.name}")

        source_size = src.total_size_bytes
        total_source_bytes += source_size

        # Calculate source bitrate
        bitrates = [f.bitrate for f in src.audio_files if f.bitrate]
        src_bitrate = sum(bitrates) // len(bitrates) if bitrates else None

        console.print(f"  Source: {src.file_count} files, {format_size(source_size)}, {format_bitrate(src_bitrate)}")
        assert src.metadata is not None  # populated above in line 332
        console.print(f"  Metadata: {src.metadata.title} by {src.metadata.author}")
        if src.metadata.series:
            console.print(f"  Series: {src.metadata.series} #{src.metadata.series_number}")

        # Show AI corrections if any
        if i - 1 in ai_corrections:
            ai = ai_corrections[i - 1]
            if ai.changes:
                console.print(f"  [yellow]AI: {', '.join(ai.changes)}[/yellow]")
            if ai.quality_note:
                console.print(f"  [yellow]⚠ {ai.quality_note}[/yellow]")

        # Convert
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(f"  Converting to M4B ({bitrate}kbps {'mono' if not stereo else 'stereo'})...", total=None)
            result = processor.process(src)
            results.append(result)

        # Handle skipped result
        if result.status == ProcessingStatus.SKIPPED:
            console.print(f"  [yellow]⊘ Skipped: {result.error_message}[/yellow]")
            console.print()
            continue

        if result.success and result.output_path:
            output_size = result.output_path.stat().st_size
            total_output_bytes += output_size
            compression = (1 - output_size / source_size) * 100 if source_size > 0 else 0
            actual_bitrate = result.ai_decision.target_bitrate_kbps if result.ai_decision else bitrate
            method = "[cyan]remuxed[/cyan]" if result.was_remuxed else f"[dim]transcoded to {actual_bitrate}kbps[/dim]"

            console.print(f"  [green]✓ Success[/green] ({method})")
            console.print(f"    Output:      {result.output_path.name}")
            console.print(f"    Size:        {format_size(output_size)} ({compression:.0f}% smaller)")
            console.print(f"    Duration:    {format_duration(result.duration_ms)}")
            console.print(f"    Chapters:    {result.chapters_preserved}")
        else:
            console.print(f"  [red]✗ Failed: {result.error_message}[/red]")

        console.print()  # Blank line between audiobooks

    # Summary
    success_count = sum(1 for r in results if r.success)
    console.print("─" * 60)
    console.print(f"[bold]Complete:[/bold] {success_count}/{len(results)} processed successfully")
    if total_source_bytes > 0 and total_output_bytes > 0:
        total_compression = (1 - total_output_bytes / total_source_bytes) * 100
        console.print(
            f"[bold]Total:[/bold] {format_size(total_source_bytes)} → {format_size(total_output_bytes)} "
            f"({total_compression:.0f}% reduction)"
        )

    print_cache_debug()


def _show_dry_run(processor, sources, output):
    """Show what would be processed in dry-run mode."""
    table = Table(title="Would Process")
    table.add_column("Source", style="cyan")
    table.add_column("Files", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Destination", style="green")

    for src in sources:
        # File info and metadata already populated, AI corrections already applied
        dest = processor.organizer.compute_destination(src.metadata, output)
        table.add_row(
            src.source_path.name,
            str(src.file_count),
            format_size(src.total_size_bytes),
            str(dest.name),
        )

    console.print(table)


@app.command()
def info(
    path: Path = typer.Argument(
        ...,
        help="Audiobook file or directory to inspect",
        exists=True,
        resolve_path=True,
    ),
    ai_verify: bool = typer.Option(False, "--ai-verify", help="Use Claude to verify metadata"),
) -> None:
    """Show detailed information about an audiobook."""
    from audiobook_optimizer.adapters.ffmpeg import FFmpegConverter
    from audiobook_optimizer.adapters.tagger import MutagenTagger

    if path.is_file():
        # Single file info
        converter = FFmpegConverter()
        tagger = MutagenTagger()

        probe = converter.probe_file(path)
        chapters = converter.extract_chapters_from_probe(probe)
        metadata = tagger.read_existing_metadata(path)

        # Get detailed format info
        file_size = path.stat().st_size
        duration_ms = converter.probe_duration(path)
        audio_stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), {})
        bitrate = int(audio_stream.get("bit_rate", 0)) // 1000 or None
        sample_rate = audio_stream.get("sample_rate")
        channels = audio_stream.get("channels")
        codec = audio_stream.get("codec_name", "unknown")

        console.print(f"\n[bold]{path.name}[/bold]")
        console.print(f"  Size:        {format_size(file_size)}")
        console.print(f"  Duration:    {format_duration(duration_ms)}")
        console.print(f"  Codec:       {codec.upper()}")
        console.print(f"  Bitrate:     {format_bitrate(bitrate)}")
        console.print(f"  Sample Rate: {sample_rate} Hz" if sample_rate else "")
        console.print(f"  Channels:    {channels} ({'stereo' if channels == 2 else 'mono' if channels == 1 else ''})")
        console.print(f"  Chapters:    {len(chapters)}")

        if metadata:
            console.print("\n  [bold]Metadata:[/bold]")
            for key, val in metadata.items():
                if val:
                    console.print(f"    {key}: {val}")

        if chapters:
            console.print("\n  [bold]Chapters:[/bold]")
            for ch in chapters[:10]:
                console.print(f"    {format_duration(ch.start_ms)} - {ch.title}")
            if len(chapters) > 10:
                console.print(f"    ... and {len(chapters) - 10} more")
    else:
        # Directory info - scan as audiobook
        processor = AudiobookProcessor(source_dir=path.parent, output_dir=Path("/tmp"))
        sources = list(processor.scanner.scan_directory(path))

        if not sources:
            console.print("[yellow]Not recognized as audiobook directory[/yellow]")
            return

        src = sources[0]
        processor._populate_file_info(src)
        inferred = processor.metadata_extractor.infer_metadata(src)

        # Calculate bitrate stats
        bitrates = [f.bitrate for f in src.audio_files if f.bitrate]
        avg_bitrate = sum(bitrates) // len(bitrates) if bitrates else None
        sample_rates = [f.sample_rate for f in src.audio_files if f.sample_rate]
        common_sample_rate = max(set(sample_rates), key=sample_rates.count) if sample_rates else None

        console.print(f"\n[bold]{src.source_path.name}[/bold]")
        console.print(f"  Files:       {src.file_count} × {src.primary_format.value.upper()}")
        console.print(f"  Total Size:  {format_size(src.total_size_bytes)}")
        console.print(f"  Duration:    {format_duration(src.total_duration_ms)}")
        console.print(f"  Avg Bitrate: {format_bitrate(avg_bitrate)}")
        if common_sample_rate:
            console.print(f"  Sample Rate: {common_sample_rate} Hz")

        console.print("\n  [bold]Inferred Metadata:[/bold]")
        console.print(f"    Title:  {inferred.title}")
        console.print(f"    Author: {inferred.author}")
        if inferred.series:
            console.print(f"    Series: {inferred.series} #{inferred.series_number}")

        # AI verification if requested
        if ai_verify:
            try:
                from audiobook_optimizer.adapters.ai_metadata import ClaudeMetadataVerifier

                verifier = ClaudeMetadataVerifier()

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task("Verifying with AI...", total=None)
                    verification = verifier.verify_metadata(src, inferred)

                console.print(f"\n  [bold]AI Verification ({verification.confidence:.0%} confidence):[/bold]")
                if verification.has_changes:
                    console.print("    [yellow]Suggested corrections:[/yellow]")
                    for change in verification.changes_made:
                        console.print(f"      • {change}")
                    console.print("\n    [bold]Corrected Metadata:[/bold]")
                    console.print(f"      Title:  {verification.suggested.title}")
                    console.print(f"      Author: {verification.suggested.author}")
                    if verification.suggested.series:
                        console.print(f"      Series: {verification.suggested.series} #{verification.suggested.series_number}")
                else:
                    console.print("    [green]Metadata looks correct[/green]")
                console.print(f"    Reasoning: {verification.reasoning}")
            except RuntimeError as e:
                console.print(f"\n  [red]AI verification failed: {e}[/red]")

    print_cache_debug()


@app.command()
def verify(
    source: Path = typer.Argument(
        ...,
        help="Directory to scan for audiobooks",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Verify metadata for all audiobooks using AI."""
    try:
        from audiobook_optimizer.adapters.ai_metadata import ClaudeMetadataVerifier

        verifier = ClaudeMetadataVerifier()
    except RuntimeError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    processor = AudiobookProcessor(source_dir=source, output_dir=Path("/tmp"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Scanning for audiobooks...", total=None)
        sources = processor.scan()

    if not sources:
        console.print("[yellow]No audiobooks found.[/yellow]")
        return

    console.print(f"Verifying metadata for [cyan]{len(sources)}[/cyan] audiobook(s)\n")

    changes_found = 0
    for src in sources:
        processor._populate_file_info(src)
        inferred = processor.metadata_extractor.infer_metadata(src)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(f"Checking: {src.source_path.name[:50]}...", total=None)
            verification = verifier.verify_metadata(src, inferred)

        if verification.has_changes:
            changes_found += 1
            console.print(f"[yellow]⚠[/yellow]  {src.source_path.name}")
            console.print(f"   Inferred: {inferred.title} by {inferred.author}")
            console.print(f"   Suggested: {verification.suggested.title} by {verification.suggested.author}")
            for change in verification.changes_made:
                console.print(f"     • {change}")
        else:
            console.print(f"[green]✓[/green]  {src.source_path.name}")

    console.print(f"\n[bold]Summary:[/bold] {changes_found}/{len(sources)} audiobooks have suggested corrections")

    print_cache_debug()


if __name__ == "__main__":
    app()
