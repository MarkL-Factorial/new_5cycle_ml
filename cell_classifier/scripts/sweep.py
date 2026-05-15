"""Thin wrapper. Prefer `cell-classifier sweep` (installed console entry)."""

import sys

from cell_classifier.cli import main

if __name__ == "__main__":
    # Insert "sweep" as the subcommand so argparse routes correctly.
    sys.exit(main(["sweep", *sys.argv[1:]]))
