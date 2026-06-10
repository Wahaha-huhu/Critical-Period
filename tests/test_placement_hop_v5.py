from cplm.injection.placement_hop_v5 import make_demo_sentences, build_placement_hop_dataset, validate_dataset


def test_placement_hop_v5_demo_builds():
    cfg = {
        "seed": 123,
        "corpus": {"globs": [], "use_demo_if_no_corpus": True, "demo_sentences": 600, "max_sentences": 600},
        "parser": {"use_spacy_for_demo": False},
        "cache": {"dir": "/tmp/cplm_test_cache_v5", "reuse": False},
        "wordhop": {"n_train": 50, "n_probe": 20, "hop_distance": 4, "marker": "<HOP>"},
        "naturalness": {"proxy_threshold": 2.0},
        "facts": {"enabled": False},
    }
    data, meta = build_placement_hop_dataset(cfg)
    assert len(data["wordhop_train"]) == 50
    assert len(data["wordhop_probe"]) == 20
    assert data["wordhop_train"][0]["marker"] == "<HOP>"
    errors, gates = validate_dataset(data, [r["source_text"] for r in data["wordhop_train"]], 2.0)
    assert not [e for e in errors if "marker-index" in e]
