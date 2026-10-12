"""Held-out checkpoint selection by penalized task objective."""

from __future__ import annotations

from typing import Any, Dict, Optional


def penalized_eval_score(
    coverage: float,
    mean_latency: float,
    mean_sample_usage: float,
    *,
    latency_penalty: float,
    parallel_penalty: float,
    terminal_bonus: float,
) -> float:
    return (
        float(terminal_bonus) * float(coverage)
        - float(latency_penalty) * float(mean_latency)
        - float(parallel_penalty) * max(float(mean_sample_usage) - 1.0, 0.0)
    )


def score_from_eval_metrics(
    metrics: Dict[str, Any],
    *,
    latency_penalty: float,
    parallel_penalty: float,
    terminal_bonus: float,
) -> float:
    return penalized_eval_score(
        metrics["mean_coverage"],
        metrics["mean_latency"],
        metrics["mean_sample_usage"],
        latency_penalty=latency_penalty,
        parallel_penalty=parallel_penalty,
        terminal_bonus=terminal_bonus,
    )


def eval_step_eligible(outer_step: int, min_step: int) -> bool:
    """Holdout snapshots before ``min_step`` cannot become ``best_eval``."""
    return int(outer_step) >= int(min_step)


def is_better_eval_checkpoint(
    score: float,
    latency: float,
    best_score: float,
    best_latency: float,
    *,
    outer_step: Optional[int] = None,
    min_step: int = 0,
) -> bool:
    if outer_step is not None and not eval_step_eligible(outer_step, min_step):
        return False
    if score > best_score + 1e-12:
        return True
    if abs(score - best_score) <= 1e-12 and latency < best_latency - 1e-12:
        return True
    return False

