from __future__ import annotations

import torch

from cplm.induction.data import InductionVocab, make_batch, query_key_position, original_value_position
from cplm.induction.model import InductionModelConfig, InductionTransformer
from cplm.induction.metrics import evaluate_recall


def test_induction_batch_positions():
    vocab = InductionVocab(n_symbols=16)
    batch = make_batch(batch_size=8, n_pairs=4, vocab=vocab, query_pair_indices=[0, 1, 2, 3])
    qpos = query_key_position(4)
    assert batch["input_ids"].shape[1] == 13
    assert torch.all(batch["query_pos"] == qpos)
    for b in range(8):
        qi = int(batch["query_pair_index"][b])
        assert int(batch["value_pos"][b]) == original_value_position(qi)
        assert int(batch["labels"][b, qpos]) == int(batch["target_value"][b])
        assert int(batch["input_ids"][b, int(batch["value_pos"][b])]) == int(batch["target_value"][b])


def test_induction_model_eval_smoke():
    vocab = InductionVocab(n_symbols=16)
    model = InductionTransformer(InductionModelConfig(vocab_size=vocab.vocab_size, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=16))
    metrics = evaluate_recall(model, vocab=vocab, n_pairs=4, batch_size=8, n_batches=1, device="cpu")
    assert "recall_accuracy" in metrics
    assert "induction_score" in metrics
    assert "acc_dist_0" in metrics

from cplm.induction.train import lr_at_step


def test_induction_lr_schedules():
    base = {"total_steps": 1000, "warmup_steps": 100, "peak_lr": 1e-3, "min_lr": 1e-5}
    assert lr_at_step(50, {**base, "schedule": "s1_decay"}) == 5e-4
    assert lr_at_step(1000, {**base, "schedule": "s1_decay"}) <= 1.1e-5
    assert lr_at_step(500, {**base, "schedule": "s2_constant", "constant_lr": 2e-4}) == 2e-4
    cyc = {**base, "schedule": "s3_cyclic", "max_lr": 3e-4, "min_lr": 3e-5, "cycle_steps": 200}
    assert lr_at_step(101, cyc) <= 3e-4
    assert lr_at_step(101, cyc) >= 3e-5
    assert lr_at_step(301, cyc) <= 3e-4
    assert lr_at_step(301, cyc) >= 3e-5
