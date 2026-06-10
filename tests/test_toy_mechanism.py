from pathlib import Path
import torch

from cplm.toy.data import sample_toy_batch, VOCAB
from cplm.toy.model import TinyCausalTransformer, ToyModelConfig
from cplm.toy.metrics import evaluate_task


def test_toy_batch_shapes():
    b = sample_toy_batch(8, task="base")
    assert b.input_ids.shape == (8, 6)
    assert b.labels.shape == (8, 6)
    assert int(b.input_ids[0, 0]) == VOCAB["<bos>"]


def test_toy_model_forward_and_eval():
    m = TinyCausalTransformer(ToyModelConfig(vocab_size=len(VOCAB), d_model=16, n_layers=1, n_heads=2, d_ff=32))
    b = sample_toy_batch(4, task="class_injection")
    logits = m(b.input_ids)
    assert logits.shape == (4, 6, len(VOCAB))
    loss = m.loss(b.input_ids, b.labels)
    assert torch.isfinite(loss)
    out = evaluate_task(m, "base")
    assert set(["accuracy", "margin", "loss"]).issubset(out)
