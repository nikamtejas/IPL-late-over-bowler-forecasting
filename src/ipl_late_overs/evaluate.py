"""Stage: evaluate. The single final evaluation on the held-out test seasons (week 6).

This is the only module allowed to open the test set, and only when run with `--final`.
"""

from __future__ import annotations

import argparse
import logging

from .splits import FinalEvaluationKey

log = logging.getLogger(__name__)


def _open_test_set() -> FinalEvaluationKey:
    """Issue the one key that `splits.test_rows` accepts. Called only from main() with --final."""
    FinalEvaluationKey._armed = True
    return FinalEvaluationKey()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final", action="store_true", help="run the one-time evaluation on the test set")
    args = ap.parse_args(argv)
    if not args.final:
        raise SystemExit("evaluate.py only runs the final test evaluation; pass --final (week 6).")
    raise SystemExit("The final evaluation is implemented in week 6; the test set stays closed until then.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
