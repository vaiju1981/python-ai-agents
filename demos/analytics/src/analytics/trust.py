"""Answer trust-grading and abstention (P0 defensibility).

Lifts ATLAS ``causal/estimator.py`` trust-tier logic into a generic, evidence-
based grade applied to any analytics answer. A result is graded
``TRUSTED`` / ``DIRECTIONAL`` / ``INSUFFICIENT`` from the evidence it rests on:
join coverage, sample size ``n``, and validation gates (A/A unbiased,
parallel-trends, etc.). Below an evidence threshold the engine is expected to
*abstain* (return INSUFFICIENT) rather than guess — the defensibility bar.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Tiers, ordered from weakest to strongest.
TIERS = ["INSUFFICIENT", "DIRECTIONAL", "TRUSTED"]

# Evidence thresholds. A join with < MIN_COVERAGE or a sample < MIN_N cannot be
# asserted as authoritative; below ABSTAIN_N we abstain outright.
MIN_COVERAGE = float(os.getenv("ANALYTICS_TRUST_MIN_COVERAGE", "0.5"))
MIN_N = int(os.getenv("ANALYTICS_TRUST_MIN_N", "30"))
ABSTAIN_N = int(os.getenv("ANALYTICS_TRUST_ABSTAIN_N", "8"))
TRUSTED_N = int(os.getenv("ANALYTICS_TRUST_TRUSTED_N", "200"))

# Safe clamp bounds so auto-tuning (PR-9) can never push a trust threshold into
# unsafe territory. ``(low, high)`` per threshold.
TRUST_BOUNDS: dict[str, tuple[float, float]] = {
    "MIN_COVERAGE": (0.1, 0.95),
    "MIN_N": (5, 200),
    "ABSTAIN_N": (3, 50),
    "TRUSTED_N": (50, 1000),
}

# Opt-in auto-calibration toggle (default off → recommend-only path).
AUTO_TUNE_ENV = "ANALYTICS_TRUST_AUTO_TUNE"


@dataclass
class TrustGrade:
    tier: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    abstain: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
            "abstain": self.abstain,
        }


def grade(
    coverage: float | None = None,
    n: int | None = None,
    gates: dict[str, bool] | None = None,
    borrowed_fraction: float = 0.0,
) -> TrustGrade:
    """Grade an answer from its evidence.

    ``coverage`` — fraction of the population the answer covers (e.g. join
    coverage). ``n`` — effective sample size. ``gates`` — passed validation
    gates (aa, parallel_trends, oos, ...). ``borrowed_fraction`` — share of the
    estimate that comes from borrowed/cross-entity evidence (caps the tier).
    """
    gates = gates or {}
    reasons: list[str] = []
    n = n or 0
    cov = coverage if coverage is not None else 1.0

    if n < ABSTAIN_N:
        return TrustGrade(
            tier="INSUFFICIENT",
            confidence=0.0,
            reasons=[f"n={n} below abstain threshold {ABSTAIN_N}"],
            abstain=True,
        )

    # Confidence is a simple monotonic blend of coverage and relative sample size.
    cov_f = max(0.0, min(1.0, cov / max(MIN_COVERAGE, 1.0)))
    n_f = max(0.0, min(1.0, n / TRUSTED_N))
    confidence = round(0.6 * cov_f + 0.4 * n_f, 4)

    passed = [k for k, v in gates.items() if v]
    all_passed = bool(gates) and len(passed) == len(gates)
    if passed:
        reasons.append(f"gates passed: {', '.join(passed)}")

    if cov < MIN_COVERAGE:
        reasons.append(f"coverage {cov:.2f} < {MIN_COVERAGE}")
        tier = "INSUFFICIENT"
    elif n >= TRUSTED_N and all_passed:
        tier = "TRUSTED"
    elif n >= MIN_N and (all_passed or passed):
        tier = "DIRECTIONAL"
    else:
        tier = "DIRECTIONAL"
        reasons.append(f"n={n} (DIRECTIONAL: needs >= {TRUSTED_N} for TRUSTED)")

    # Borrowed evidence may inform the estimate but never raises the tier.
    if borrowed_fraction >= 0.5 or n < 3:
        tier = _min_tier(tier, "DIRECTIONAL")
        reasons.append("borrowed/cross evidence caps tier at DIRECTIONAL")

    return TrustGrade(
        tier=tier, confidence=confidence, reasons=reasons, abstain=tier == "INSUFFICIENT"
    )


def _min_tier(a: str, b: str) -> str:
    return a if TIERS.index(a) <= TIERS.index(b) else b


def should_abstain(grade: TrustGrade) -> bool:
    """True when the engine must refuse to answer rather than guess."""
    return grade.abstain or grade.tier == "INSUFFICIENT"


def abstain_message(grade: TrustGrade, what: str = "this question") -> str:
    return (
        f"[ABSTAIN] Insufficient evidence to answer {what} defensibly "
        f"(trust={grade.tier}, reasons={grade.reasons}). "
        f"I will not guess; gather more data or a stronger join."
    )


# --- PR-9: opt-in auto-calibration of trust thresholds from labeled outcomes ---


def current_thresholds() -> dict[str, float]:
    """Snapshot of the live trust thresholds (mutable at runtime via tuning)."""
    return {
        "MIN_COVERAGE": MIN_COVERAGE,
        "MIN_N": MIN_N,
        "ABSTAIN_N": ABSTAIN_N,
        "TRUSTED_N": TRUSTED_N,
    }


def _clamp(name: str, value: float) -> tuple[float, bool]:
    lo, hi = TRUST_BOUNDS[name]
    v = max(lo, min(hi, value))
    return v, (v != value)


def apply_trust_tuning(action: str, *, step: float = 1.25) -> dict[str, Any]:
    """Apply a suggested ``raise``/``lower`` to the live trust thresholds.

    ``raise`` makes the bar *stricter* (thresholds grow, so it is harder to
    reach a higher tier); ``lower`` makes it *looser* (thresholds shrink, so
    lower tiers can be promoted). Every value is clamped to ``TRUST_BOUNDS`` so
    auto-tuning can never move a threshold past its safe limit. ``hold``/``keep``
    are no-ops. Returns ``{"action", "old", "new", "changed", "clamped"}``.

    The thresholds are module globals that :func:`grade` reads at call time, so
    mutating them here changes grading live without touching caller code.
    """
    action = (action or "hold").lower()
    if action not in ("raise", "lower"):
        snap = current_thresholds()
        return {"action": action, "old": snap, "new": snap, "changed": False, "clamped": []}
    factor = step if action == "raise" else 1.0 / step
    old = current_thresholds()
    new = dict(old)
    clamped: list[str] = []
    for name in ("MIN_COVERAGE", "MIN_N", "ABSTAIN_N", "TRUSTED_N"):
        v, hit = _clamp(name, old[name] * factor)
        if hit:
            clamped.append(name)
        new[name] = v
    for name, v in new.items():
        globals()[name] = v
    changed = any(new[n] != old[n] for n in old)
    return {"action": action, "old": old, "new": new, "changed": changed, "clamped": clamped}


def _tuning_sample_count(suggestion: dict[str, Any]) -> int:
    by_tier = suggestion.get("samplesByTier") or {}
    if by_tier:
        return sum(int(v) for v in by_tier.values())
    sample = suggestion.get("sample")
    if isinstance(sample, int):
        return sample
    # audit_store-style suggestion reports rates, not counts: treat as unknown.
    return 0


def auto_tune_trust_thresholds(
    suggestion: dict[str, Any],
    *,
    min_samples: int = 30,
    enabled: bool | None = None,
    audit: Any = None,
) -> dict[str, Any]:
    """Opt-in self-calibration from a threshold-tuning ``suggestion``.

    Gated by ``ANALYTICS_TRUST_AUTO_TUNE`` (default off → recommend-only path,
    the existing behavior). When enabled and the suggested action is
    ``raise``/``lower`` with enough labeled evidence (``>= min_samples``), the
    change is applied via :func:`apply_trust_tuning` and an audit entry
    recording old→new plus the evidence is written to ``audit`` (an
    ``SqliteAuditStore`` or anything with ``record_trust_tuning``).
    """
    if enabled is None:
        enabled = os.getenv(AUTO_TUNE_ENV, "0") == "1"
    if not enabled:
        return {"applied": False, "reason": f"{AUTO_TUNE_ENV} not enabled (recommend-only)"}
    action = (suggestion.get("action") or "keep").lower()
    if action in ("hold", "keep", "none"):
        return {"applied": False, "reason": "no change suggested"}
    samples = _tuning_sample_count(suggestion)
    if samples < min_samples:
        return {
            "applied": False,
            "reason": f"labeled outcomes {samples} < min_samples {min_samples}",
        }
    result = apply_trust_tuning(action)
    if not result["changed"]:
        return {
            "applied": False,
            "reason": "clamped — no net change",
            "old": result["old"],
            "new": result["new"],
        }
    if audit is not None:
        writer = getattr(audit, "record_trust_tuning", None)
        if callable(writer):
            writer(result["old"], result["new"], {"suggestion": suggestion, "samples": samples})
    return {"applied": True, "samples": samples, **result}
