from homework2_rent import score_rent


def test_rent():
    assert score_rent() >= 0.48
