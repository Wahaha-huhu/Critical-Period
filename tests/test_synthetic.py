from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cplm.data import ProbeBank, SyntheticAgreementGenerator, Vocab, build_probe_items


def make_gen(onset=10):
    vocab = Vocab.build(20, 10, 3)
    bank = ProbeBank(vocab, 16, 16, seed=123)
    return SyntheticAgreementGenerator(
        vocab,
        bank,
        run_seed=0,
        onset_step=onset,
        mix_before={"local": 0.5, "pp_same": 0.5, "pp_opp": 0.0},
        mix_after={"local": 0.35, "pp_same": 0.35, "pp_opp": 0.30},
    )


def test_generator_deterministic():
    g = make_gen()
    a = g.generate_sentence(12, 3)
    b = g.generate_sentence(12, 3)
    assert a == b


def test_no_pp_opp_before_onset_for_many_samples():
    g = make_gen(onset=100)
    seen = {g.choose_template(0, i) for i in range(100)}
    assert "pp_opp" not in seen


def test_probe_has_three_templates_per_item():
    g = make_gen()
    items = build_probe_items(g, "val")
    by_id = {}
    for item in items:
        by_id.setdefault(item["item_id"], set()).add(item["template"])
    assert all(v == {"local", "pp_same", "pp_opp"} for v in by_id.values())
