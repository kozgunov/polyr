"""Local Polybot package bootstrap and version metadata."""

from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.1.0"

# Operational configuration intentionally remains in the user-visible project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
