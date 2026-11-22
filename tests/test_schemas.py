import pandas as pd
import pytest

from ipl_late_overs.schemas import Col, Schema, SchemaError

S = Schema(
    name="t",
    key=("id",),
    columns=(Col("id", "str"), Col("n", "int", min=0, max=10), Col("kind", "str", nullable=True, allowed=("a", "b"))),
)


def good():
    return S.coerce(pd.DataFrame({"id": ["x", "y"], "n": [1, 2], "kind": ["a", None]}))


def test_valid_frame_passes():
    S.validate(good())


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda df: df.assign(n=pd.array([1, None], dtype="Int64")), "nulls"),
        (lambda df: df.assign(n=pd.array([1, 11], dtype="Int64")), "above"),
        (lambda df: df.assign(n=pd.array([-1, 1], dtype="Int64")), "below"),
        (lambda df: df.assign(kind=pd.array(["a", "c"], dtype="string")), "unexpected values"),
        (lambda df: df.assign(id=pd.array(["x", "x"], dtype="string")), "duplicate"),
        (lambda df: df.assign(n=["1", "2"]), "expected int"),
        (lambda df: df.assign(extra=1), "unexpected columns"),
        (lambda df: df.drop(columns="kind"), "missing columns"),
    ],
)
def test_invalid_frames_fail_loudly(mutate, msg):
    with pytest.raises(SchemaError, match=msg):
        S.validate(mutate(good()))
