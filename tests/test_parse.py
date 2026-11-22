"""Parse fixture matches and check them against the published scorecards."""

import pytest

from ipl_late_overs import parse, schemas

from .conftest import load_fixture

# (match_id, innings) -> (team, runs, wickets, legal balls), from the ESPNcricinfo scorecards.
SCORECARDS = {
    # IPL 2017 match 1, Hyderabad: SRH 207/4 (20), RCB 172 all out (19.4). SRH won by 35 runs.
    ("1082591", 1): ("Sunrisers Hyderabad", 207, 4, 120),
    ("1082591", 2): ("Royal Challengers Bangalore", 172, 10, 118),
    # IPL 2023 match 1, Ahmedabad: CSK 178/7 (20), GT 182/5 (19.2). GT won by 5 wickets.
    ("1359475", 1): ("Chennai Super Kings", 178, 7, 120),
    ("1359475", 2): ("Gujarat Titans", 182, 5, 116),
    # IPL 2017, Rajkot, super over: MI 11/2 (two wickets end it, after 5 balls), GL 6/0. MI won.
    ("1082625", 3): ("Mumbai Indians", 11, 2, 5),
    ("1082625", 4): ("Gujarat Lions", 6, 0, 6),
}


def innings_summary(deliveries, match_id, innings):
    d = deliveries[(deliveries["match_id"] == match_id) & (deliveries["innings"] == innings)]
    return (d["batting_team"].iloc[0], int(d["runs_total"].sum()), int(d["n_wickets"].sum()), int(d["is_legal"].sum()))


@pytest.mark.parametrize("key", sorted(SCORECARDS))
def test_innings_totals_match_scorecard(fixture_tables, key):
    team, runs, wickets, legal = SCORECARDS[key]
    got = innings_summary(fixture_tables["deliveries"], *key)
    assert got == (team, runs, wickets, legal)


def test_match_metadata(fixture_tables):
    m = fixture_tables["matches"].set_index("match_id")
    r = m.loc["1082591"]
    assert (r["season"], r["season_label"], str(r["date"].date())) == (2017, "2017", "2017-04-05")
    assert (r["winner"], r["win_by_runs"], r["target_runs"], r["target_overs"]) == ("Sunrisers Hyderabad", 35, 208, 20.0)
    assert not r["has_super_over"] and r["first_innings_team"] == "Sunrisers Hyderabad"

    gt = m.loc["1359475"]
    assert (gt["winner"], gt["win_by_wickets"], gt["target_runs"]) == ("Gujarat Titans", 5, 179)


def test_target_equals_first_innings_plus_one(fixture_tables):
    m = fixture_tables["matches"].set_index("match_id")
    for mid in m.index[m["method"].isna()]:
        first = innings_summary(fixture_tables["deliveries"], mid, 1)[1]
        assert m.loc[mid, "target_runs"] == first + 1


def test_super_over_flags(fixture_tables):
    d = fixture_tables["deliveries"]
    so = d[d["match_id"] == "1082625"].groupby("innings")["is_super_over"].agg(["all", "any"])
    assert list(so["all"]) == [False, False, True, True] and list(so["any"]) == [False, False, True, True]
    m = fixture_tables["matches"].set_index("match_id").loc["1082625"]
    assert m["has_super_over"] and m["result"] == "tie" and m["eliminator"] == "Mumbai Indians"
    assert m["target_runs"] == 154  # target of the regular second innings, not the super over


def test_impact_player_squads_and_replacements(fixture_tables):
    mp = fixture_tables["match_players"]
    sizes = mp[mp["match_id"] == "1359475"].groupby("team").size()
    assert set(sizes) == {12}
    rep = fixture_tables["replacements"]
    imp = rep[(rep["match_id"] == "1359475") & (rep["kind"] == "match")]
    assert len(imp) == 2 and set(imp["reason"]) == {"impact_player"}
    assert set(imp["player_in_id"]) <= set(mp["player_id"])  # impact subs are in info.players


def test_every_player_has_a_registry_id(fixture_tables):
    d = fixture_tables["deliveries"]
    j = load_fixture("1082591")["info"]["registry"]["people"]
    sub = d[d["match_id"] == "1082591"]
    assert (sub["bowler"].map(j) == sub["bowler_id"]).all()
    assert (sub["batter"].map(j) == sub["batter_id"]).all()


def test_delivery_ordering(fixture_tables):
    d = fixture_tables["deliveries"]
    for _, g in d.groupby(["match_id", "innings"]):
        assert list(g["ball_seq"]) == list(range(1, len(g) + 1))
        assert g["over"].is_monotonic_increasing


def test_extras_and_legality(fixture_tables):
    d = fixture_tables["deliveries"]
    extras = d[[f"extras_{k}" for k in parse.EXTRAS_KINDS]].sum(axis=1)
    assert (extras == d["runs_extras"]).all()
    assert (d["runs_batter"] + d["runs_extras"] == d["runs_total"]).all()
    assert (d["is_legal"] == ((d["extras_wides"] == 0) & (d["extras_noballs"] == 0))).all()


def test_tables_satisfy_schemas(fixture_tables):
    for name, schema in parse.OUTPUTS.items():
        schema.validate(fixture_tables[name])


def test_runs_that_do_not_add_up_fail_loudly():
    j = load_fixture("1082591")
    j["innings"][0]["overs"][0]["deliveries"][0]["runs"]["total"] += 1
    with pytest.raises(parse.ParseError, match="don't add up"):
        parse.parse_match("1082591", j)


def test_unknown_player_fails_loudly():
    j = load_fixture("1082591")
    j["innings"][0]["overs"][0]["deliveries"][0]["bowler"] = "Not A Player"
    with pytest.raises(parse.ParseError, match="no registry ID"):
        parse.parse_match("1082591", j)


def test_parquet_roundtrip(fixture_tables, tmp_path):
    p = tmp_path / "deliveries.parquet"
    written = schemas.write_table(fixture_tables["deliveries"], schemas.DELIVERIES, p)
    back = schemas.read_table(p, schemas.DELIVERIES)
    assert back.equals(written)
