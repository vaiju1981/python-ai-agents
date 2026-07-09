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
            tier="INSUFFICIENT", confidence=0.0,
            reasons=[f"n={n} below abstain threshold {ABSTAIN_N}"], abstain=True,
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

    return TrustGrade(tier=tier, confidence=confidence, reasons=reasons,
                      abstain=tier == "INSUFFICIENT")


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
