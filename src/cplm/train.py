from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import ProbeBank, SyntheticAgreementGenerator, Vocab, build_probe_items
from .eval import evaluate_probe
from .models import build_model
from .results import ResultManager
from .schedules import lr_at_step, set_optimizer_lr


def select_device(config: dict[str, Any]) -> torch.device:
    requested = str(config["experiment"].get("device", "cuda"))
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        requested = "cpu"
    return torch.device(requested)


def maybe_autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    from contextlib import nullcontext

    return nullcontext()


def build_common_objects(config: dict[str, Any], run_seed: int, onset_step: int, device: torch.device):
    d = config["data"]
    vocab = Vocab.build(int(d["n_nouns"]), int(d["n_verbs"]), int(d["n_preps"]))
    bank = ProbeBank(
        vocab,
        val_size=int(d["val_probe_size"]),
        test_size=int(d["test_probe_size"]),
        seed=int(d["probe_seed"]),
    )
    generator = SyntheticAgreementGenerator(
        vocab=vocab,
        probe_bank=bank,
        run_seed=run_seed,
        onset_step=onset_step,
        mix_before=d["mix_before"],
        mix_after=d["mix_after"],
    )
    model = build_model(config, vocab_size=len(vocab.id_to_token)).to(device)
    return vocab, bank, generator, model


def train_run(
    config: dict[str, Any],
    result_manager: ResultManager,
    schedule: str,
    seed: int,
    onset_step: int,
    max_steps: int | None = None,
    target_pp_opp_count: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train one run.

    If target_pp_opp_count is supplied, the loop continues past `total_steps`
    until the actual PP-opp count reaches the target, or until `max_steps`.
    """
    result_manager.prepare(overwrite=overwrite)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = select_device(config)
    if device.type == "cpu":
        torch.set_num_threads(int(config["experiment"].get("cpu_threads", 1)))
    precision = str(config["experiment"].get("precision", "bf16"))
    vocab, bank, generator, model = build_common_objects(config, seed, onset_step, device)
    result_manager.append_jsonl("vocab.jsonl", vocab.to_jsonable())

    tconf = config["train"]
    dconf = config["data"]
    total_steps = int(tconf["total_steps"])
    if max_steps is None:
        max_steps = total_steps
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tconf["peak_lr"]),
        betas=tuple(float(x) for x in tconf["betas"]),
        eps=float(tconf["eps"]),
        weight_decay=float(tconf["weight_decay"]),
    )
    if bool(tconf.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    checkpoint_steps = config["experiment"].get("checkpoint_steps", [])
    if checkpoint_steps == "auto_log":
        from .config import auto_checkpoint_steps

        checkpoint_steps = auto_checkpoint_steps(total_steps)
    checkpoint_steps = set(int(s) for s in checkpoint_steps)
    eval_interval = int(config["experiment"].get("eval_interval", 500))
    log_interval = int(config["experiment"].get("log_interval", 50))
    progress_to_stdout = bool(config["experiment"].get("progress_to_stdout", True))
    progress_interval = int(config["experiment"].get("progress_interval", eval_interval))
    batch_size = int(dconf["batch_size"])
    seq_len = int(dconf["sequence_length"])
    grad_clip = float(tconf.get("grad_clip", 0.0))

    val_items = build_probe_items(generator, "val")
    test_items = build_probe_items(generator, "test")

    counts_total = {"local": 0, "pp_same": 0, "pp_opp": 0, "sentences": 0, "tokens": 0}
    lr_weighted_pp_opp = 0.0
    started = time.time()
    last_loss = float("nan")
    ema_loss = float("nan")

    def emit_eval(step: int, split: str = "val") -> dict[str, float]:
        items = val_items if split == "val" else test_items
        metrics = evaluate_probe(model, items, pad_id=vocab.pad_id, device=device)
        payload = {
            "step": step,
            "split": split,
            "schedule": schedule,
            "seed": seed,
            "onset_step": onset_step,
            "loss_last": last_loss,
            **metrics,
            **{f"count_{k}": v for k, v in counts_total.items()},
            "lr_weighted_pp_opp": lr_weighted_pp_opp,
        }
        result_manager.append_jsonl("metrics_eval.jsonl", payload)
        if progress_to_stdout and split == "val":
            print(
                f"EVAL run={result_manager.spec.run_id} step={step} "
                f"loss_last={last_loss:.4f} local={metrics.get('acc_local', float('nan')):.3f} "
                f"pp_same={metrics.get('acc_pp_same', float('nan')):.3f} "
                f"pp_opp={metrics.get('acc_pp_opp', float('nan')):.3f} "
                f"inv={metrics.get('invariance', float('nan')):.3f} "
                f"pp_opp_count={counts_total['pp_opp']}",
                flush=True,
            )
        return metrics

    # Initial evaluation and checkpoint.
    emit_eval(0, "val")
    if 0 in checkpoint_steps and bool(config["experiment"].get("save_checkpoints", True)):
        result_manager.save_checkpoint(0, {"model": model.state_dict(), "optimizer": opt.state_dict(), "counts": counts_total})

    step = 0
    while step < max_steps:
        step += 1
        lr = lr_at_step(
            step=step,
            schedule=schedule,
            total_steps=total_steps,
            peak_lr=float(tconf["peak_lr"]),
            warmup_fraction=float(tconf["warmup_fraction"]),
            terminal_lr_factor=float(tconf["terminal_lr_factor"]),
        )
        set_optimizer_lr(opt, lr)
        input_ids, labels, counts = generator.make_batch(step, batch_size, seq_len, device=device)
        opt.zero_grad(set_to_none=True)
        with maybe_autocast(device, precision):
            out = model(input_ids, labels=labels)
            loss = out["loss"]
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        last_loss = float(loss.detach().cpu())
        ema_loss = last_loss if ema_loss != ema_loss else 0.98 * ema_loss + 0.02 * last_loss
        for k in counts_total:
            counts_total[k] += int(counts[k])
        lr_weighted_pp_opp += float(lr) * int(counts["pp_opp"])

        elapsed = time.time() - started
        tokens_per_sec = counts_total["tokens"] / elapsed if elapsed > 0 else float("nan")
        steps_per_sec = step / elapsed if elapsed > 0 else float("nan")

        if step % log_interval == 0 or step == 1:
            result_manager.append_jsonl(
                "metrics_train.jsonl",
                {
                    "step": step,
                    "loss": last_loss,
                    "ema_loss": ema_loss,
                    "lr": lr,
                    "schedule": schedule,
                    "seed": seed,
                    "onset_step": onset_step,
                    "readout": result_manager.spec.readout,
                    "elapsed_sec": elapsed,
                    "steps_per_sec": steps_per_sec,
                    "tokens_per_sec": tokens_per_sec,
                    "count_local": counts_total["local"],
                    "count_pp_same": counts_total["pp_same"],
                    "count_pp_opp": counts_total["pp_opp"],
                    "count_sentences": counts_total["sentences"],
                    "count_tokens": counts_total["tokens"],
                    "lr_weighted_pp_opp": lr_weighted_pp_opp,
                },
            )
            result_manager.append_jsonl(
                "counts.jsonl",
                {
                    "step": step,
                    "lr": lr,
                    "local": counts_total["local"],
                    "pp_same": counts_total["pp_same"],
                    "pp_opp": counts_total["pp_opp"],
                    "sentences": counts_total["sentences"],
                    "tokens": counts_total["tokens"],
                    "lr_weighted_pp_opp": lr_weighted_pp_opp,
                },
            )
        if progress_to_stdout and (step % progress_interval == 0 or step == 1):
            print(
                f"TRAIN run={result_manager.spec.run_id} step={step}/{max_steps} "
                f"loss={last_loss:.4f} ema={ema_loss:.4f} lr={lr:.2e} "
                f"pp_opp_count={counts_total['pp_opp']} tok/s={tokens_per_sec:.0f}",
                flush=True,
            )
        if step % eval_interval == 0 or step == total_steps:
            emit_eval(step, "val")
        if step in checkpoint_steps and bool(config["experiment"].get("save_checkpoints", True)):
            result_manager.save_checkpoint(
                step,
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "counts": counts_total,
                    "lr_weighted_pp_opp": lr_weighted_pp_opp,
                    "step": step,
                },
            )
        if target_pp_opp_count is not None and step >= total_steps and counts_total["pp_opp"] >= target_pp_opp_count:
            break

    val_metrics = emit_eval(step, "val")
    test_metrics = emit_eval(step, "test")
    final = {
        "step": step,
        "schedule": schedule,
        "seed": seed,
        "onset_step": onset_step,
        "target_pp_opp_count": target_pp_opp_count,
        "dose_matched": target_pp_opp_count is None or counts_total["pp_opp"] >= target_pp_opp_count,
        "counts": counts_total,
        "lr_weighted_pp_opp": lr_weighted_pp_opp,
        "val": val_metrics,
        "test": test_metrics,
        "elapsed_sec": time.time() - started,
    }
    ResultManager._write_json(result_manager.run_dir / "final.json", final)
    return final
