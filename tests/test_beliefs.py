from agora.beliefs import BeliefBoard, parse_date_ord


def _board():
    b = BeliefBoard()
    k = BeliefBoard.make_key("Meridian Labs", "ceo")
    return b, k


def test_supersession_by_date():
    b, k = _board()
    assert b.assert_fact(k, "Chen", parse_date_ord("Jan 2025")) == "new"
    assert b.assert_fact(k, "Okafor", parse_date_ord("May 2025")) == "superseded"
    assert b.current(k).value == "okafor"
    assert b.n_supersessions(k) == 1


def test_stale_echo_raises_doubt_not_value():
    b, k = _board()
    b.assert_fact(k, "Chen", parse_date_ord("Jan 2025"))
    b.assert_fact(k, "Okafor", parse_date_ord("May 2025"))
    doubt_before = b.doubt(k)
    assert b.assert_fact(k, "Chen", parse_date_ord("Mar 2025")) == "stale-echo"
    assert b.current(k).value == "okafor"  # echo never wins
    assert b.doubt(k) > doubt_before


def test_doubt_monotone_in_churn():
    b1, k = _board()
    b1.assert_fact(k, "Chen", parse_date_ord("Jan 2025"))
    b2 = BeliefBoard()
    b2.assert_fact(k, "Chen", parse_date_ord("Jan 2025"))
    b2.assert_fact(k, "Okafor", parse_date_ord("Mar 2025"))
    b2.assert_fact(k, "Larsson", parse_date_ord("May 2025"))
    # More observed churn on the same key => less trust in the current value.
    assert b2.doubt(k) > b1.doubt(k)


def test_refresh_same_value():
    b, k = _board()
    b.assert_fact(k, "Chen", parse_date_ord("Jan 2025"))
    assert b.assert_fact(k, "Chen", parse_date_ord("Jun 2025")) == "refresh"
    assert b.n_supersessions(k) == 0


def test_date_parse_orders():
    assert parse_date_ord("Mar 2025") < parse_date_ord("Nov 2025") < parse_date_ord("Jan 2026")
    assert parse_date_ord("no date here") == 0


def test_missing_key():
    b, _ = _board()
    assert b.doubt("nope::x") == 1.0
    assert b.current("nope::x") is None
