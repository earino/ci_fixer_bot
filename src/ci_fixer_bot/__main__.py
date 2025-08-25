"""
Entry point for ci_fixer_bot when run as python -m ci_fixer_bot
"""

if __name__ == "__main__":
    import sys
    from ci_fixer_bot.cli import main
    sys.exit(main())