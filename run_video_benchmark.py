#!/usr/bin/env python3
"""
run_video_benchmark.py
======================
Entry point for the CashVision Continuous Handheld Video Stream Benchmark.
Executes the proposed Adaptive Cascade expert pipeline across the 36 continuous
real-world video streams (Contribution C3 of the CashVision ESWA manuscript).

For full details, CLI flags, and implementation, see `run_c3.py`.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_c3 import main

if __name__ == "__main__":
    main()
