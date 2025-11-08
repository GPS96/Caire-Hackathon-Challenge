"""Main entry point for CAIRE arrhythmia detection training."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arrhythmia_project.train import main

if __name__ == "__main__":
    main()
