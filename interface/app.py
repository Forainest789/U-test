"""Thin Streamlit entrypoint for the pairwise preference app.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pairwise_preference_app import main


if __name__ == "__main__":
    main()
