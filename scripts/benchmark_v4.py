#!/usr/bin/env python3
"""Measure the same CPU benchmark as v3, using the v4 artifact."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_v3 import main

if __name__ == '__main__':
    main('v4')
