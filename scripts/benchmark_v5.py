#!/usr/bin/env python3
"""CPU benchmark for v5, including full-size local thumbnail measurements."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_v3 import main

if __name__ == '__main__':
    main('v5')
