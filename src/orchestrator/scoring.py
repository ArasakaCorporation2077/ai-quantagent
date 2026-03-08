"""Selection and weighting helpers for OOS-first alpha ranking."""


def compute_oos_score(
    sharpe_oos: float | None,
    n_oos: int | None,
    target_n_oos: int = 60,
) -> float:
    """Score an alpha by OOS Sharpe with a small-sample penalty."""
    if sharpe_oos is None or sharpe_oos <= 0:
        return 0.0

    if target_n_oos <= 0:
        sample_factor = 1.0
    else:
        sample_factor = min(1.0, max(float(n_oos or 0), 0.0) / float(target_n_oos))

    return float(sharpe_oos) * sample_factor
