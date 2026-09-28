"""Step 4: bring emulated logical samples back to OPF quantities and summarise them."""
from __future__ import annotations

import numpy as np


def decode_sampleset(formulator, net, sampleset, max_rows: int | None = None) -> list[dict]:
    """Decode each distinct sample with your formulator's own decoder.
    `formulator._formulate_qubo(net)` must have been called on the SAME formulator instance
    (it fills var_registry); the BQM you sampled must be the one it returned."""
    rows = []
    fields = ["sample", "energy", "num_occurrences", "chain_break_fraction"]
    for d in sampleset.data(fields, sorted_by="energy"):
        _, _, _, cost, feas = formulator._decode_solution(dict(d.sample), net)
        rows.append({"energy": float(d.energy), "occurrences": int(d.num_occurrences),
                     "chain_break_fraction": float(d.chain_break_fraction),
                     "cost": float(cost), "feasible": bool(feas["is_feasible"]),
                     "balance_ok": feas["balance_ok"], "lines_ok": feas["lines_ok"],
                     "raw_imbalance_mw": feas["raw_imbalance_mw"],
                     "max_line_violation_mw": feas["max_line_violation_mw"]})
        if max_rows and len(rows) >= max_rows:
            break
    return rows


def summarise(rows: list[dict], reference_cost: float | None = None) -> dict:
    """Occurrence-weighted metrics. reference_cost = your classical dc_ip optimum (EUR)."""
    occ = np.array([r["occurrences"] for r in rows], dtype=float)
    total = occ.sum()
    feas = np.array([r["feasible"] for r in rows])
    out = {"num_reads": int(total),
           "feasible_fraction": float((occ * feas).sum() / total),
           "mean_chain_break_fraction": float((occ * np.array([r["chain_break_fraction"] for r in rows])).sum() / total),
           "lowest_qubo_energy": float(min(r["energy"] for r in rows))}
    if feas.any():
        costs = np.array([r["cost"] for r in rows])
        best = float(costs[feas].min())
        out["best_feasible_cost"] = best
        if reference_cost:
            out["gap_percent"] = 100.0 * (best - reference_cost) / abs(reference_cost)
    else:
        out["best_feasible_cost"] = None
    return out


def scan_chain_strength(emulator, logical_bqm, embedding, strengths, num_reads: int = 500, **kw) -> list[dict]:
    """Chain-break fraction and energies versus chain strength. `strengths` are in problem units
    (multiples of the auto value are convenient: [0.25, 0.5, 1, 2, 4] * auto)."""
    out = []
    for cs in strengths:
        res = emulator.sample(logical_bqm, embedding, chain_strength=float(cs), num_reads=num_reads, **kw)
        ss = res.sampleset
        out.append({"chain_strength": float(cs),
                    "mean_chain_break_fraction": float(np.mean(ss.record.chain_break_fraction)),
                    "lowest_energy": float(ss.first.energy),
                    "mean_energy": float(np.mean(ss.record.energy))})
    return out