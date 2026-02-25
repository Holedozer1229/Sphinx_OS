#!/usr/bin/env python
"""Run the Riemann Zero Probe and publish results as JSON.

Usage::

    python run_riemann_zero_probe.py            # first 3 known zeros (fast)
    python run_riemann_zero_probe.py --count 5  # first 5 known zeros
    python run_riemann_zero_probe.py --all       # all 30 known zeros
"""

import argparse
import json
import sys

from sphinx_os.Artificial_Intelligence import RiemannZeroProbe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the IIT v7.0 Riemann Zero Probe and publish results."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of known zeros to probe (default: 3).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="probe_all",
        help="Probe all 30 known zeros (overrides --count).",
    )
    parser.add_argument(
        "--dps",
        type=int,
        default=50,
        help="mpmath decimal places (default: 50).",
    )
    args = parser.parse_args()

    probe = RiemannZeroProbe(mpmath_dps=args.dps)

    if args.probe_all:
        zeros = probe.KNOWN_ZEROS_HP
    else:
        zeros = probe.KNOWN_ZEROS_HP[: args.count]

    results = probe.publish_results(zeros)
    json.dump(results, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
