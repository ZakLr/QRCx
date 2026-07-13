"""Terminal summary table formatting."""
import pandas as pd


def print_summary(results):
    """Pretty-print experiment results."""
    results.summary()
