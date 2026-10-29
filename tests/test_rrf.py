from app.retrieval import reciprocal_rank_fusion, tokenize


def test_rrf_ranks_items_seen_in_both_lists_higher():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]], k=60)
    scores = dict(fused)
    assert scores[1] > scores[3]
    assert scores[2] > scores[4]


def test_rrf_returns_empty_for_empty_rankings():
    assert reciprocal_rank_fusion([[], []]) == []


def test_tokenize_lowercases_and_splits_text():
    assert tokenize("MRI and CT") == ["mri", "and", "ct"]
