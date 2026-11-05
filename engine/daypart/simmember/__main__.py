"""A6 — ``python -m daypart.simmember --seed 0 --out <dir>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from daypart.simmember.cohort import generate
from daypart.simmember.manifest import write


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m daypart.simmember", description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = write(generate(args.seed), args.out)
    print(
        f"wrote {manifest['counts']['curated']} curated + {manifest['counts']['synthetic']}"
        f" synthetic profiles to {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
