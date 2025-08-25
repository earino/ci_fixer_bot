"""
Command-line interface for ci_fixer_bot.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from .config import Config, load_config
from .core import CIFixerBot

console = Console()


@click.command()
@click.argument("repository_url")
@click.option(
    "--analyze-runs",
    default=5,
    help="Number of recent CI runs to analyze",
    show_default=True,
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview issues without creating them",
)
@click.option(
    "--focus",
    type=click.Choice(["safe", "tactical", "strategic", "all"]),
    default="all",
    help="Focus on specific issue types",
    show_default=True,
)
@click.option(
    "--min-priority",
    type=int,
    default=1,
    help="Only create issues with priority >= N (1-100)",
    show_default=True,
)
@click.option(
    "--llm-provider",
    type=click.Choice(["claude-cli", "openai", "ollama", "custom-endpoint", "custom-cli", "none"]),
    help="Override LLM provider from config",
)
@click.option(
    "--llm-model",
    help="Specify model for LLM provider (e.g., gpt-4, codellama)"
)
@click.option(
    "--llm-endpoint",
    help="Custom endpoint URL for LLM provider"
)
@click.option(
    "--vector-store-type",
    type=click.Choice(["memory", "faiss"], case_sensitive=False),
    default="memory",
    help="Vector store type (memory=default, faiss=special cases with millions of issues)"
)
@click.option(
    "--auto-calibrate",
    is_flag=True,
    help="Auto-calibrate similarity threshold based on existing issues"
)
@click.option(
    "--calibration-strategy",
    type=click.Choice(["conservative", "balanced", "aggressive"], case_sensitive=False),
    default="balanced",
    help="Strategy for auto-calibration (conservative=high precision, balanced=best F1, aggressive=high recall)"
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output results as JSON",
)
@click.version_option()
def main(
    repository_url: str,
    analyze_runs: int,
    dry_run: bool,
    focus: str,
    min_priority: int,
    llm_provider: Optional[str],
    llm_model: Optional[str],
    llm_endpoint: Optional[str],
    vector_store_type: str,
    auto_calibrate: bool,
    calibration_strategy: str,
    config_path: Optional[Path],
    verbose: bool,
    json_output: bool,
) -> None:
    """
    Analyze CI failures and create intelligent GitHub issues.
    
    REPOSITORY_URL: GitHub repository URL (https://github.com/owner/repo)
    
    Examples:
    
        # Basic analysis
        ci_fixer_bot https://github.com/owner/repo
        
        # Preview without creating issues
        ci_fixer_bot https://github.com/owner/repo --dry-run
        
        # Focus on safe fixes only
        ci_fixer_bot https://github.com/owner/repo --focus=safe
        
        # Use OpenAI instead of default LLM
        ci_fixer_bot https://github.com/owner/repo --llm-provider=openai
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Override config with CLI arguments
        if llm_provider:
            config.llm.provider = llm_provider
        if llm_model:
            config.llm.model = llm_model
        if llm_endpoint:
            config.llm.endpoint = llm_endpoint
        
        # Set auto-calibration options
        if auto_calibrate:
            config.deduplication.embedding.auto_calibrate = True
            config.deduplication.embedding.calibration_strategy = calibration_strategy
        
        if verbose:
            console.print(f"🤖 [bold blue]ci_fixer_bot[/] analyzing {repository_url}")
            console.print(f"📋 Configuration: {focus} issues, {analyze_runs} runs, LLM: {config.llm.provider}")
            if auto_calibrate:
                console.print(f"🎯 Auto-calibrating threshold with {calibration_strategy} strategy")
        
        # Initialize the bot
        bot = CIFixerBot(config=config, verbose=verbose)
        
        # Run analysis
        results = bot.analyze_repository(
            repository_url=repository_url,
            analyze_runs=analyze_runs,
            focus=focus,
            min_priority=min_priority,
            dry_run=dry_run,
        )
        
        if json_output:
            import json
            print(json.dumps(results.to_dict(), indent=2))
            return
        
        # Display results
        _display_results(results, dry_run, verbose)
        
    except KeyboardInterrupt:
        console.print("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ Error: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _display_results(results, dry_run: bool, verbose: bool) -> None:
    """Display analysis results in a nice format."""
    
    if not results.issues:
        console.print("✅ [green]No CI issues found![/] Your repository is in great shape.")
        return
    
    console.print(f"\n📊 [bold]Analysis Complete[/] - Found {len(results.issues)} issues")
    
    # Summary table
    table = Table(title="Issue Summary by Risk Level")
    table.add_column("Risk Level", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Description", style="italic")
    
    safe_count = len([i for i in results.issues if i.risk_level == "safe"])
    tactical_count = len([i for i in results.issues if i.risk_level == "tactical"])
    strategic_count = len([i for i in results.issues if i.risk_level == "strategic"])
    
    table.add_row("🟢 Safe", str(safe_count), "Can be fixed by anyone")
    table.add_row("🟡 Tactical", str(tactical_count), "Needs testing, low risk")
    table.add_row("🔴 Strategic", str(strategic_count), "Requires planning, high risk")
    
    console.print(table)
    
    if dry_run:
        console.print("\n👁️  [yellow]DRY RUN[/] - Issues would be created:")
    else:
        console.print("\n🚀 [green]Issues created successfully:[/]")
    
    # List issues
    for i, issue in enumerate(results.issues, 1):
        risk_emoji = {"safe": "🟢", "tactical": "🟡", "strategic": "🔴"}[issue.risk_level]
        
        console.print(f"\n{i}. {risk_emoji} {issue.title}")
        console.print(f"   Effort: {issue.estimated_effort} | Priority: {issue.priority}")
        
        if not dry_run and hasattr(issue, 'github_url'):
            console.print(f"   🔗 {issue.github_url}")
    
    if not dry_run:
        repo_issues_url = results.repository_url.replace('github.com', 'github.com').rstrip('.git') + '/issues'
        console.print(f"\n✨ [bold green]All done![/] View issues: {repo_issues_url}")
    
    # Statistics
    if verbose and results.stats:
        console.print(f"\n📈 [bold]Analysis Stats:[/]")
        console.print(f"   CI runs analyzed: {results.stats.get('ci_runs_analyzed', 0)}")
        console.print(f"   Total failures found: {results.stats.get('total_failures', 0)}")
        console.print(f"   Patterns identified: {results.stats.get('patterns_found', 0)}")
        console.print(f"   Analysis time: {results.stats.get('analysis_time_seconds', 0):.1f}s")


if __name__ == "__main__":
    main()