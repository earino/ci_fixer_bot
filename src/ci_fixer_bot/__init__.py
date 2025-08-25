"""
ci_fixer_bot: Intelligent CI failure analysis tool

This package provides intelligent analysis of CI failures and creates
risk-aware, actionable GitHub issues.
"""

__version__ = "0.1.0"
__author__ = "Eduardo Ariño de la Rubia"
__email__ = "earino@gmail.com"

from .cli import main

__all__ = ["main"]