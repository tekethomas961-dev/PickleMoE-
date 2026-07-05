from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pickleball_moe.demo import generate_demo_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic pickleball shot CSV for smoke tests.")
    parser.add_argument("--out", default="data/demo_pickleball.csv", help="Output CSV path.")
    parser.add_argument("--n-samples", type=int, default=600, help="Number of synthetic rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    path = generate_demo_csv(args.out, n_samples=args.n_samples, seed=args.seed)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
