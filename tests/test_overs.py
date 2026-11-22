"""Late-over table: five overs rebuilt by hand from the raw JSON, plus rule and leakage checks."""

import pytest

from ipl_late_overs import config, overs, parse

from .conftest import FIXTURES, load_fixture

# Rebuilt by hand from the fixture JSON: per-over runs of the chase summed up to the over,
# wicket entries counted (none were "retired hurt"), bowlers read from each over's first ball.
# None of these innings has a miscounted over, so legal balls before = 6 per over.
# Every result agrees with the ESPNcricinfo scorecard of that final.
HAND_BUILT = {
    # 2008 final: RR need 8 off the last over from Balaji, win off the last ball (a wide in the over).
    ("336040", 19): dict(target=164, runs_scored=156, runs_needed=8, wickets_fallen=7, balls_remaining=6,
                         striker="Sohail Tanvir", non_striker="SK Warne", bowler="L Balaji", prev_bowler="M Ntini",
                         bowler_overs_before=3, runs_in_over=8, legal_balls_in_over=6, deliveries_in_over=7,
                         over_completed=True, chase_ended_in_over=True),
    # 2014 final: KKR need 5 off 6, Chawla hits the winning four off the 3rd ball from Awana.
    ("734049", 19): dict(target=200, runs_scored=195, runs_needed=5, wickets_fallen=7, balls_remaining=6,
                         striker="SP Narine", non_striker="PP Chawla", bowler="P Awana", prev_bowler="MG Johnson",
                         bowler_overs_before=3, runs_in_over=5, legal_balls_in_over=3, deliveries_in_over=3,
                         over_completed=False, chase_ended_in_over=True),
    # 2017 final: RPS need 11 off Johnson's last over, make 9 and lose 3 wickets. MI win by 1 run.
    ("1082650", 19): dict(target=130, runs_scored=119, runs_needed=11, wickets_fallen=3, balls_remaining=6,
                          striker="MK Tiwary", non_striker="SPD Smith", bowler="MG Johnson", prev_bowler="JJ Bumrah",
                          bowler_overs_before=3, runs_in_over=9, legal_balls_in_over=6, deliveries_in_over=6,
                          over_completed=True, chase_ended_in_over=True),
    # 2019 final: CSK need 9 off Malinga's last over, make 7. MI win by 1 run.
    ("1181768", 19): dict(target=150, runs_scored=141, runs_needed=9, wickets_fallen=5, balls_remaining=6,
                          striker="SR Watson", non_striker="RA Jadeja", bowler="SL Malinga", prev_bowler="JJ Bumrah",
                          bowler_overs_before=3, runs_in_over=7, legal_balls_in_over=6, deliveries_in_over=6,
                          over_completed=True, chase_ended_in_over=True),
    # 2022 final: GT need 4 off 12; Gill hits McCoy's first ball for six.
    ("1312200", 18): dict(target=131, runs_scored=127, runs_needed=4, wickets_fallen=3, balls_remaining=12,
                          striker="Shubman Gill", non_striker="DA Miller", bowler="OC McCoy", prev_bowler="R Ashwin",
                          bowler_overs_before=3, runs_in_over=6, legal_balls_in_over=1, deliveries_in_over=1,
                          over_completed=False, chase_ended_in_over=True),
}


def build(tables):
    kept, chase, excl = overs.select_chases(tables["matches"], tables["deliveries"])
    late, loads = overs.build_late_overs(kept, chase)
    return kept, chase, excl, late, loads


@pytest.fixture(scope="module")
def built(fixture_tables):
    return build(fixture_tables)


def name_to_id(match_id):
    return load_fixture(match_id)["info"]["registry"]["people"]


def test_verification_overs_match_config():
    cfg = config.load_config()["late_overs"]["verification_overs"]
    assert {(v["match_id"], v["over"]) for v in cfg} == set(HAND_BUILT)


@pytest.mark.parametrize("key", sorted(HAND_BUILT))
def test_hand_built_over(built, key):
    late = built[3].set_index(["match_id", "over"])
    row = late.loc[key]
    exp = HAND_BUILT[key]
    ids = name_to_id(key[0])
    assert row["target"] == exp["target"]
    assert row["runs_scored"] == exp["runs_scored"]
    assert row["runs_needed"] == exp["runs_needed"]
    assert row["wickets_fallen"] == exp["wickets_fallen"]
    assert row["wickets_in_hand"] == 10 - exp["wickets_fallen"]
    assert row["balls_remaining"] == exp["balls_remaining"]
    assert row["legal_balls_bowled"] == 6 * key[1]
    assert row["required_rate"] == pytest.approx(6 * exp["runs_needed"] / exp["balls_remaining"])
    assert row["striker_id"] == ids[exp["striker"]]
    assert row["non_striker_id"] == ids[exp["non_striker"]]
    assert row["bowler_id"] == ids[exp["bowler"]]
    assert row["prev_bowler_id"] == ids[exp["prev_bowler"]]
    assert row["bowler_overs_before"] == exp["bowler_overs_before"]
    assert row["bowler_legal_balls_before"] == 6 * exp["bowler_overs_before"]
    for col in ("runs_in_over", "legal_balls_in_over", "deliveries_in_over", "over_completed", "chase_ended_in_over"):
        assert row[col] == exp[col], col


def test_exclusions(built):
    kept, _, excl, late, _ = built
    by_rule = {r["rule"]: r["match_ids"] for r in excl["match_rules"]}
    assert by_rule["outcome method present (D/L)"] == ["1370353"]  # 2023 final, rain-reduced
    assert "1370353" not in set(kept["match_id"])
    assert "1370353" not in set(late["match_id"])
    assert set(late["over"]) <= set(range(14, 20))


def test_super_over_not_in_table(built):
    # 1082625 was tied: the regular chase is kept, the super over is not.
    late = built[3]
    tie = late[late["match_id"] == "1082625"]
    assert sorted(tie["over"]) == [14, 15, 16, 17, 18, 19]
    assert tie["runs_in_over"].sum() + tie["runs_scored"].iloc[0] == 153


def test_every_started_late_over_is_a_row(built):
    _, chase, _, late, _ = built
    src = chase[chase["over"].between(14, 19)][["match_id", "over"]].drop_duplicates()
    assert len(src) == len(late)


def test_table_checks_pass(built):
    _, chase, _, late, loads = built
    overs.check_late_overs(late, loads, chase)


def test_state_ignores_balls_of_the_over_itself(fixture_tables):
    """Changing what happens IN an over must not change that over's pre-over state."""
    j = load_fixture("1082650")
    over19 = next(o for o in j["innings"][1]["overs"] if o["over"] == 19)
    d = over19["deliveries"][0]
    d["runs"] = {"batter": 6, "extras": 0, "total": 6}
    tables = parse.tables_from_matches([parse.parse_match("1082650", j)])
    late = build(tables)[3].set_index("over")
    assert late.loc[19, "runs_scored"] == 119 and late.loc[19, "runs_in_over"] == 11


def test_retired_hurt_is_not_a_dismissal_but_retired_out_is():
    j = load_fixture("1082650")
    # 3 wickets fell before over 19. Relabel the first as retired hurt, then as retired out.
    first_w = next(d for o in j["innings"][1]["overs"] for d in o["deliveries"] if "wickets" in d)
    first_w["wickets"][0]["kind"] = "retired hurt"
    hurt = build(parse.tables_from_matches([parse.parse_match("1082650", j)]))[3].set_index("over")
    first_w["wickets"][0]["kind"] = "retired out"
    out = build(parse.tables_from_matches([parse.parse_match("1082650", j)]))[3].set_index("over")
    assert hurt.loc[19, "wickets_fallen"] == 2
    assert out.loc[19, "wickets_fallen"] == 3


def test_bowler_loads(built):
    loads = built[4].set_index(["match_id", "over"])
    ids = name_to_id("1181768")
    before_last = loads.loc[("1181768", 19)].set_index("bowler_id")
    # MI bowlers before the 2019 final's last over: Bumrah 4 (just bowled the 19th), Malinga 3.
    assert before_last.loc[ids["JJ Bumrah"], "overs_bowled"] == 4
    assert before_last.loc[ids["JJ Bumrah"], "bowled_previous_over"]
    assert before_last.loc[ids["SL Malinga"], "overs_bowled"] == 3
    assert before_last["bowled_previous_over"].sum() == 1
