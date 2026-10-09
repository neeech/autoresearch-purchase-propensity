"""Run from the repository root: python autoresearch.py --experiments 30."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from autoresearch_purchase_propensity.research import main

if __name__ == '__main__':
    main()
