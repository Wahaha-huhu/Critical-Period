from cplm.injection.wordhop import SourceSentence, transform_sentence, tokenize_simple
from cplm.injection.scoring import logprob_margin, placement_selectivity


def test_wordhop_skips_punctuation():
    tokens = tokenize_simple("She paints, slowly and carefully, every small wooden figurine.")
    src = SourceSentence(
        source_id="x",
        tokens=tokens,
        verb_index=tokens.index("paints"),
        verb_inflected="paints",
        verb_lemma="paint",
        marker="S",
        subject_number="singular",
        split="probe",
        template="punctuation",
        has_attractor=False,
    )
    rec = transform_sentence(src, "WORDHOP", hop_distance=4)
    assert rec["tokens"][rec["marker_index"]] == "S"
    # Words after verb, skipping commas: slowly, and, carefully, every.
    assert rec["tokens"][rec["marker_index"] - 1] == "every"
    assert rec["tokens"][rec["marker_index"] + 1] == "small"


def test_nohop_immediate_after_lemma():
    tokens = tokenize_simple("The author writes detailed notes in her journal.")
    src = SourceSentence("x", tokens, tokens.index("writes"), "writes", "write", "S", "singular", "probe", "plain", False)
    rec = transform_sentence(src, "NOHOP")
    assert rec["tokens"][rec["verb_index_transformed"]] == "write"
    assert rec["marker_index"] == rec["verb_index_transformed"] + 1
    assert rec["text"].startswith("The author write S")


def test_logprob_margin():
    score = logprob_margin({"S": -0.5, "P": -1.5}, "S", "P")
    assert score.correct
    assert score.margin == 1.0


def test_placement_selectivity():
    assert placement_selectivity(-0.1, [-3.0, -4.0]) > 0
