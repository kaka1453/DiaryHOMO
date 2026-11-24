"""Compatibility entry point for the current product command."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diary_core.infer.batch import main


if __name__ == "__main__":
    main()
