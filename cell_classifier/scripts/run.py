"""Thin wrapper. Prefer `cell-classifier` (installed console entry)."""

import sys

from cell_classifier.cli import main

if __name__ == "__main__":
    sys.exit(main())
