"""Kill-fast decision scoreboard across the whole counterfactual-utility line.

The other modules each emit ONE report (``e0.json``, ``intervention_contract.json``,
``utility_report.json``). This module reads those reports and applies the six decisive
questions from ``docs/validation-matrix.md``, printing a single PASS / BLOCK / PENDING
table with reason codes. It duplicates no logic: it aggregates verdicts that upstream
steps already computed, so the project has one "may I keep spending GPU?" entry point.

Order is the point (see ``utest/README.md``). Each gate is a one-way block: a BLOCK stops
the method line and names the fallback in ``docs/research-plan.md`` §18; a PASS is not
evidence, it only clears the way to the next gate. A PENDING gate has not run yet and
must not be skipped -- later gates are uninterpretable without it.

    python -m utest.gates \
        --e0 runs/e0.json \
        --intervention-contract runs/events/story_017/arms/intervention_contract.json \
        --utility-report runs/events/story_017/arms/utility_report.json \
        --signals-json domain_signals.json \
        --out gates.json

``--signals-json`` supplies the three domain verdicts no module can produce on its own
(frozen in W2/W4, not inferred from data)::

    {
      "ruler_range_usable": true,   # D2.5: ceiling-vs-noise range has resolution
      "content_causal": true,       # M2: correct separates from matched wrong, >=10/12
      "predictable": false          # M4: held-out calibration beats relevance baselines
    }

Exit code is 1 when any gate BLOCKs, 0 otherwise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

MIN_VIABLE_STORIES = 128

# (id, question, cheapest_test, fallback) in kill-fast order. `fallback` names the §18
# retreat that a BLOCK routes to.
GATE_ORDER = (
    ("E0", "estimand exists", "eligibility audit (zero GPU)",
     "R: add a second training source before any controller claim"),
    ("Q2", "intervention actually happened", "fixed-prefix five-arm contract (M1)",
     "repair harness; do not touch the model"),
    ("Q3", "metric ruler has range", "D2.5 ceiling-vs-noise range (zero GPU)",
     "replace or demote the identity/quality component at W2"),
    ("Q1", "content channel exists", "correct vs matched wrong on dev-M2 (M2)",
     "R2 content-causal audit, no controller"),
    ("Q4", "decoded utility is heterogeneous", "correct vs none pilot (M3)",
     "R3 all-memory dominates -> no estimand; no-memory dominates -> checkpoint mismatch"),
    ("Q5", "utility is predictable from cheap features", "two-tower MLP held-out (M4)",
     "R4 keep the online teacher, drop the deployed router"),
)


def evaluate(signals: Mapping) -> list[dict]:
    """Apply the six decisive questions to a flat ``signals`` dict.

    Each signal maps to one gate. ``None`` (or missing) means the gate has not run yet
    and is reported PENDING rather than passed; a missing signal must never be read as a
    pass, because every later gate is uninterpretable without it.
    """
    results: list[dict] = []

    def row(gate_id: str, signal: str, ok: bool, *, evidence=None):
        value = signals.get(signal, None)
        status = "pending" if value is None else ("pass" if bool(value) else "block")
        reason = None if status == "pass" else (
            f"{signal}_not_supplied" if status == "pending" else f"{signal}_false"
        )
        return {"id": gate_id, "signal": signal, "status": status,
                "reason": reason, "evidence": evidence}

    n_e = signals.get("n_eligible_stories", None)
    results.append({
        "id": "E0", "signal": "n_eligible_stories",
        "status": "pending" if n_e is None else ("pass" if int(n_e) >= MIN_VIABLE_STORIES else "block"),
        "reason": None if n_e is None else (
            None if int(n_e) >= MIN_VIABLE_STORIES
            else f"n_eligible_stories_{n_e}_below_{MIN_VIABLE_STORIES}"
        ),
        "evidence": {"n_eligible_stories": n_e},
    })

    helpful_n = signals.get("helpful_present", None)
    harmful_n = signals.get("harmful_present", None)
    heterogeneous = None
    if helpful_n is not None and harmful_n is not None:
        heterogeneous = bool(helpful_n) and bool(harmful_n)
    results.extend([
        row("Q2", "intervention_contract_valid",
            signals.get("intervention_contract_valid"),
            evidence=signals.get("intervention_evidence")),
        row("Q3", "ruler_range_usable", signals.get("ruler_range_usable")),
        row("Q1", "content_causal", signals.get("content_causal")),
        {
            "id": "Q4", "signal": "heterogeneity", "status": "pending" if heterogeneous is None else ("pass" if heterogeneous else "block"),
            "reason": None if heterogeneous is None else (None if heterogeneous else "no_heterogeneity"),
            "evidence": {"helpful_present": helpful_n, "harmful_present": harmful_n},
        },
        row("Q5", "predictable", signals.get("predictable")),
    ])
    for result, (gid, question, cheapest, fallback) in zip(results, GATE_ORDER):
        result["question"] = question
        result["cheapest_test"] = cheapest
        result["fallback_on_block"] = fallback
        result["gate"] = gid
    return results


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pop(utility_report: Mapping) -> Mapping:
    populations = utility_report.get("populations", {}) if isinstance(utility_report, dict) else {}
    return populations.get("all_eligible", {}) if isinstance(populations, dict) else {}


def signals_from_artifacts(
    e0: Mapping,
    intervention_contract: Mapping,
    utility_report: Mapping,
    domain: Mapping | None = None,
) -> dict:
    """Derive the machine-decidable signals from the three upstream reports.

    ``domain`` carries the three verdicts no report computes (ruler range, content
    causality, predictability); they must be frozen by a human at W2/W4, not inferred.
    """
    signals: dict = {
        "n_eligible_stories": e0.get("n_eligible_stories"),
        "intervention_contract_valid": (
            None if intervention_contract.get("status") is None
            else intervention_contract.get("status") == "passed"
        ),
        "intervention_evidence": {
            "errors": intervention_contract.get("errors", []),
            "decoded_l1": intervention_contract.get("decoded_l1", {}),
        },
        # Domain verdicts are always present but None until a human freezes them at W2/W4.
        "ruler_range_usable": None,
        "content_causal": None,
        "predictable": None,
    }
    if utility_report.get("populations"):
        pop = _pop(utility_report)
        helpful = pop.get("helpful_rate", {}) or {}
        harmful = pop.get("harmful_rate", {}) or {}
        signals["helpful_present"] = int(helpful.get("n", 0) or 0) > 0
        signals["harmful_present"] = int(harmful.get("n", 0) or 0) > 0
        signals["estimand"] = utility_report.get("estimand")
        signals["content_causal"] = utility_report.get("content_causal")
    else:
        signals["helpful_present"] = None
        signals["harmful_present"] = None
        signals["content_causal"] = utility_report.get("content_causal")
    if domain:
        for key in ("ruler_range_usable", "content_causal", "predictable"):
            if key in domain:
                signals[key] = domain[key]
    return signals


def _self_check() -> None:
    # A signal set with every gate passing must yield no block.
    good = {
        "n_eligible_stories": 160,
        "intervention_contract_valid": True,
        "ruler_range_usable": True,
        "content_causal": True,
        "helpful_present": True,
        "harmful_present": True,
        "predictable": True,
    }
    verdicts = {result["id"]: result["status"] for result in evaluate(good)}
    assert verdicts == {gid: "pass" for gid, *_ in GATE_ORDER}, verdicts
    assert all(result["reason"] is None for result in evaluate(good))

    # Missing signals are PENDING, never PASS.
    pending = evaluate({})
    assert all(result["status"] == "pending" for result in pending), pending

    # Each BLOCK names a fallback and never smuggles PENDING through.
    blocked = evaluate({**good, "content_causal": False, "predictable": None})
    by_id = {result["id"]: result for result in blocked}
    assert by_id["Q1"]["status"] == "block" and by_id["Q1"]["fallback_on_block"]
    assert by_id["Q5"]["status"] == "pending"

    # E0 is numeric: below the floor is a block even when supplied.
    under = evaluate({**good, "n_eligible_stories": 100})
    assert {r["id"]: r["status"] for r in under}["E0"] == "block"

    # signals_from_artifacts derives the census signal from the utility report shape.
    e0 = {"n_eligible_stories": 160}
    contract = {"status": "passed", "errors": [], "decoded_l1": {}}
    util = {"estimand": "memory_utility", "content_causal": True,
            "populations": {"all_eligible": {
                "helpful_rate": {"n": 3}, "harmful_rate": {"n": 2}}}}
    signals = signals_from_artifacts(e0, contract, util)
    assert signals["helpful_present"] and signals["harmful_present"]
    assert signals["content_causal"] is True
    assert signals["ruler_range_usable"] is None  # must come from the domain verdict
    print("[gates] self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e0", type=Path, help="eligibility report (e0.json)")
    parser.add_argument("--intervention-contract", type=Path,
                        help="intervention_contract.json from a fixed-prefix event run")
    parser.add_argument("--utility-report", type=Path,
                        help="utility_report.json (may be measurement_incomplete)")
    parser.add_argument("--signals-json", type=Path,
                        help="domain verdicts: ruler_range_usable, content_causal, predictable")
    parser.add_argument("--signals", type=Path,
                        help="raw signals dict, skipping artifact derivation")
    parser.add_argument("--out", type=Path, help="write the scoreboard as JSON")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        _self_check()
        return 0

    if args.signals:
        signals = _read_json(args.signals)
    else:
        e0 = _read_json(args.e0) if args.e0 and args.e0.is_file() else {}
        contract = (
            _read_json(args.intervention_contract)
            if args.intervention_contract and args.intervention_contract.is_file()
            else {}
        )
        util = (
            _read_json(args.utility_report)
            if args.utility_report and args.utility_report.is_file()
            else {}
        )
        domain = _read_json(args.signals_json) if args.signals_json and args.signals_json.is_file() else {}
        signals = signals_from_artifacts(e0, contract, util, domain)

    results = evaluate(signals)
    summary = {
        "blocking": [r for r in results if r["status"] == "block"],
        "pending": [r for r in results if r["status"] == "pending"],
        "gates": results,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    for result in results:
        flag = {"pass": "PASS", "block": "BLOCK", "pending": "PENDING"}[result["status"]]
        reason = f"  <- {result['reason']}" if result["reason"] else ""
        print(f"[{flag}] {result['id']}: {result['question']}{reason}")
    if summary["blocking"]:
        print(f"\nBLOCKED by {[r['id'] for r in summary['blocking']]}. "
              "Fallbacks are named in each gate's fallback_on_block.")
        return 1
    print("\nNo blocking gate. Proceed to the next gate in order; PENDING gates must run first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
