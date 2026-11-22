import json
from pathlib import Path

import pytest

from ipl_late_overs import parse

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(match_id: str) -> dict:
    return json.loads((FIXTURES / f"{match_id}.json").read_text())


@pytest.fixture(scope="session")
def fixture_tables():
    """Parsed tables for all fixture matches (runs parse's consistency checks too)."""
    ids = sorted(p.stem for p in FIXTURES.glob("*.json"))
    return parse.tables_from_matches([parse.parse_match(m, load_fixture(m)) for m in ids])
