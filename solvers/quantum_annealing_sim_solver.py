import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import numpy as np
from solvers.base_qubo_formulator import BaseQuboFormulator

try:
    import openjij as oj
except ImportError:
    raise ImportError("OpenJij is required for SQA. Run: pip install openjij")

try:
    from dwave.samplers import SteepestDescentSolver
    _HAVE_GREEDY = True
except ImportError:
    _HAVE_GREEDY = False


class SimulatedQuantumAnnealingSolver(BaseQuboFormulator):
    """
    SQA (path-integral Monte Carlo on the transverse-field Ising model, via OpenJij's SQASampler)
    applied to the DC-OPF QUBO built by BaseQuboFormulator.

    Changes from the previous version, and why
    --------------------------------------------
    2. Sampling is still a per-read Python loop, each call given its own explicit seed drawn from
       a numpy Generator seeded once by `self.seed` -- THIS PART OF THE ORIGINAL CODE WAS ALREADY
       CORRECT and is kept. Two things were tried and rejected while fixing this file, noted here
       so they aren't "fixed" again by a future edit:
         - Passing a single `seed=` to `sample(..., num_reads=N)` with N > 1: in the installed
           OpenJij version this makes every read IDENTICAL (confirmed: energy_std_dev was exactly
           0.0 across reads) -- this alone was enough to explain "huge mismatches", since 500
           supposedly-independent reads were really 1 read repeated 500 times.
         - Seeding only numpy's global RNG once and calling sample(..., num_reads=N, seed=None):
           looked reproducible on a trivial 2-variable toy problem, but is NOT reproducible on the
           real (61+ variable) case5 QUBO -- confirmed by running it twice with an identical numpy
           seed and getting different energies both times. The toy problem's "reproducibility" was
           a false positive: a 2-variable problem has so few reachable states that unrelated seeds
           can land on the same energy by chance, not because the RNG was actually controlled.
       Measured overhead of the per-read loop vs. one batched call, same total reads/sweeps: no
       measurable difference (looping is not the performance problem here; premature freezing and
       an unnormalized coefficient range are).

    3. The BQM given to the sampler is normalized (divided by its largest absolute coefficient,
       so max|coefficient| = 1) before sampling; energies are reported back in EUR by undoing the
       scale, directly comparable to `neal` (classical SA) runs on the unnormalized BQM.
       *** Normalization alone is NOT a fix. *** This QUBO's coefficients span several orders of
       magnitude (bounded-coefficient radix weights range from `mw_precision` up to roughly half
       an asset's dispatch range, and the line-limit penalty squares those weights), so a single
       global scale factor crushes the small, meaningful coefficients toward zero relative to beta
       while leaving the few huge ones dominant. `dynamic_range` (max/min nonzero |coefficient| of
       the unnormalized BQM) is reported in `sampler_stats` so this can be monitored; a large value
       (case5 at mw_precision=1.0 is ~1.8e6) is a sign the *problem* itself needs a smaller
       dynamic range (larger `mw_precision`, tighter asset/line bounds), not just better solver
       tuning. NOTE the tolerance coupling: this formulator's default `feasibility_tol_mw` equals
       `mw_precision`, so coarsening `mw_precision` also loosens what counts as feasible -- pass
       `feasibility_tol_mw` explicitly to hold the feasibility bar fixed while exploring precision.

    4. beta/gamma are chosen so that beta*gamma/trotter = O(1) at the start of a (now log-spaced,
       see point 5) schedule, instead of a beta so large that tanh(beta*gamma*(1-s)/trotter)
       saturates near s=0 and the transverse (quantum) term is effectively off for the whole run.

    5. A custom, log-spaced `schedule` replaces OpenJij's default quartic one, so an equal share of
       sweeps is spent at each energy decade -- similar to what `neal`'s default geometric beta
       schedule does.

    6. Optional greedy (steepest-descent) post-processing of every SQA read via dwave-samplers'
       SteepestDescentSolver, run on the ORIGINAL (unnormalized) bqm -- exact local descent on the
       real energy landscape, essentially free next to the SQA sampling itself. `sampler_stats`
       keeps both the raw best and the post-greedy best so you can see how much of the remaining
       gap was just local roughness (closed by greedy) versus SQA landing in the wrong basin
       entirely (not closed by greedy).

    7. `trotter_spread`: the gap between the best and worst classical-energy Trotter slice of the
       best read, in normalized units. NOTE: response.info['trotter_energies'] (as documented in
       SQASampler._get_result's own docstring) is NOT where this ends up in the installed version
       once reads are looped -- each single-read Response stores it at
       response.info['system'][0]['trotter_energies']. A large spread at s=1 means the Trotter
       slices never agreed by the end of the schedule -- a sign to run more sweeps or a slower
       schedule, not just a different seed.

    8. `optimality_gap_percent` is computed from `reference_cost_eur_per_hr` (e.g. from a
       `pandapower.rundcopp()` run on the same net) if supplied, and only when the decoded solution
       is feasible -- comparing costs of an infeasible dispatch to a feasible reference isn't
       meaningful. It is left as None otherwise, rather than hardcoded to 0.0.
    """

    def __init__(self, formulation="dc_ptdf", num_reads=300, num_sweeps=5000, max_time=1800,
                 mw_precision=1.0, trotter_slices=32, beta=8.0, gamma=1.0, schedule_points=40, schedule_s_min=1e-4,
                 postprocess_greedy=True, seed=None, **kwargs):
        """
        beta, gamma          : SQA parameters applied to the NORMALIZED bqm (max|coefficient|=1).
                               beta=8, gamma=1, trotter=32 -> beta*gamma/trotter = 0.25 at s=0, so
                               the transverse coupling is live at the start and freezes out only in
                               roughly the last third of the schedule.
        schedule_points/_s_min: log-spaced `s` schedule with this many points from schedule_s_min
                               to 1.0; num_sweeps split evenly across the points.
        postprocess_greedy    : run dwave-samplers' SteepestDescentSolver on every SQA read.
                               Silently falls back to raw-only if dwave-samplers isn't installed
                               (check `sampler_stats["greedy_postprocess_applied"]`).
        seed                  : seeds a numpy Generator that draws one independent per-read seed
                               for each of the `num_reads` calls to OpenJij -- reproducible AND
                               diverse (see point 2). None -> a fresh random batch every call.
        Any BaseQuboFormulator kwarg (feasibility_tol_mw, penalty_balance, penalty_line, ...) can
        be passed through.
        """
        super().__init__(
            formulation=formulation, max_time=max_time, mw_precision=mw_precision, **kwargs)

        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.trotter_slices = trotter_slices
        self.beta = beta
        self.gamma = gamma
        self.schedule_points = schedule_points
        self.schedule_s_min = schedule_s_min
        self.postprocess_greedy = postprocess_greedy and _HAVE_GREEDY
        self.seed = seed

        self.sampler = oj.SQASampler()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(bqm):
        """normalized_bqm = bqm / scale, scale = largest |linear or quadratic coefficient| (the
        offset is excluded: it only shifts energies, it never drives the Monte Carlo dynamics)."""
        mags = [abs(v) for v in bqm.linear.values()] + [abs(v) for v in bqm.quadratic.values()]
        scale = max(mags) if mags else 1.0
        if scale <= 0.0:
            scale = 1.0
        norm = bqm.copy()
        norm.scale(1.0 / scale)
        return norm, scale, mags

    @staticmethod
    def _dynamic_range(mags):
        nonzero = [m for m in mags if m > 1e-12]
        return float(max(nonzero) / min(nonzero)) if nonzero else 1.0

    def _log_schedule(self, num_sweeps):
        """List of (s, one_mc_step) pairs: log-spaced s in [schedule_s_min, 1], beta fixed by the
        caller (self.beta), sweeps split evenly across schedule_points stops."""
        s_values = np.geomspace(self.schedule_s_min, 1.0, self.schedule_points)
        s_values[-1] = 1.0
        per_point = max(1, num_sweeps // self.schedule_points)
        return [(float(s), int(per_point)) for s in s_values]

    @staticmethod
    def _trotter_spread(response):
        """Gap between the best and worst classical-energy Trotter slice, in NORMALIZED energy
        units, for a single-read Response (see class docstring point 7 for where this lives)."""
        systems = response.info.get('system')
        if not systems:
            return None
        slice_energies = systems[0].get('trotter_energies')
        if not slice_energies:
            return None
        return float(max(slice_energies) - min(slice_energies))

    # ------------------------------------------------------------------ #
    def solve(self, net, reference_cost_eur_per_hr=None):
        start_time = time.time()
        status, cost, dispatch, sgen_dispatch, slack_dispatch, feasibility = "Success", None, {}, {}, {}, {}
        form_time, samp_time, pp_time = 0.0, 0.0, 0.0
        bqm, complexity, sampler_stats = None, {}, {}

        try:
            # 1. Formulation Phase (CPU) via BaseQuboFormulator
            t0 = time.time()
            bqm, complexity = self._formulate_qubo(net)
            form_time = time.time() - t0

            # 2. Normalize (see class docstring point 3) and build the log-spaced schedule.
            norm_bqm, scale, mags = self._normalize(bqm)
            dyn_range = self._dynamic_range(mags)
            schedule = self._log_schedule(self.num_sweeps)

            # 3. Sample: one call per read, each with its own seed drawn from a seeded numpy
            #    Generator (see class docstring point 2 for why this loop is kept).
            rng = np.random.default_rng(self.seed)
            t1 = time.time()
            raw_samples, raw_energies, spreads = [], [], []
            for _ in range(self.num_reads):
                read_seed = int(rng.integers(0, 2**31 - 1))
                response = self.sampler.sample(
                    norm_bqm, beta=self.beta, gamma=self.gamma, trotter=self.trotter_slices,
                    schedule=schedule, num_reads=1, seed=read_seed,
                )
                raw_samples.append(response.first.sample)
                raw_energies.append(scale * response.first.energy)   # undo normalization -> EUR
                spreads.append(self._trotter_spread(response))
            samp_time = time.time() - t1

            raw_energies = np.array(raw_energies, dtype=float)
            best_raw_idx = int(np.argmin(raw_energies))
            best_raw_sample = raw_samples[best_raw_idx]
            best_raw_energy = float(raw_energies[best_raw_idx])
            unique_states = {frozenset(s.items()) for s in raw_samples}
            spread = spreads[best_raw_idx]

            # 4. Optional greedy post-processing (on the ORIGINAL, unnormalized bqm: exact descent).
            best_sample, best_energy, polish_gain = best_raw_sample, best_raw_energy, 0.0
            if self.postprocess_greedy:
                t2 = time.time()
                greedy = SteepestDescentSolver().sample(bqm, initial_states=raw_samples)
                pp_time = time.time() - t2
                g_best = greedy.first
                if g_best.energy < best_raw_energy:
                    best_sample, best_energy = g_best.sample, float(g_best.energy)
                    polish_gain = best_raw_energy - best_energy

            sampler_stats = {
                "num_reads_requested": self.num_reads,
                "num_sweeps_per_read": self.num_sweeps,
                "trotter_slices": self.trotter_slices,
                "beta": self.beta,
                "gamma": self.gamma,
                "schedule": "log-spaced custom",
                "dynamic_range": round(dyn_range, 2),
                "unique_states_found": len(unique_states),
                "energy_best_raw": round(best_raw_energy, 2),
                "energy_mean_raw": round(float(raw_energies.mean()), 2),
                "energy_worst_raw": round(float(raw_energies.max()), 2),
                "energy_std_dev_raw": round(float(raw_energies.std()), 2),
                "trotter_spread_best_read_normalized": None if spread is None else round(spread, 4),
                "greedy_postprocess_applied": self.postprocess_greedy,
                "greedy_polish_gain_eur": round(float(polish_gain), 2),
                "energy_best_after_greedy": round(float(best_energy), 2),
            }

            # 5. Decoding Phase (CPU) via BaseQuboFormulator
            dispatch, sgen_dispatch, slack_dispatch, cost, feasibility = self._decode_solution(best_sample, net)

        except Exception as e:
            status = f"Failed: {str(e)}"
            print(f"\nCRITICAL FORMULATION ERROR: {e}\n")
            raise

        exec_time = round(time.time() - start_time, 4)

        optimality_gap_percent = None
        if reference_cost_eur_per_hr and feasibility.get("is_feasible"):
            optimality_gap_percent = round(
                100.0 * (cost - reference_cost_eur_per_hr) / reference_cost_eur_per_hr, 4)

        # 6. Final Solution and Metadata Assembly
        solution = {
            "cost_eur_per_hr": cost,
            "generator_dispatch_mw": dispatch,
            "static_generator_dispatch_mw": sgen_dispatch,
            "slack_dispatch_mw": slack_dispatch,
            "grid_state": {
                "max_line_loading_percent": None,
                "max_voltage_pu": 1.0 if self.formulation.startswith("dc") else None,
                "min_voltage_pu": 1.0 if self.formulation.startswith("dc") else None,
                "feasibility": feasibility
            }
        }

        metadata = {
            "status": status,
            "solver_name": f"openjij_sqa_{self.formulation}",
            "execution_time_seconds": {
                "total": exec_time,
                "formulation_cpu": round(form_time, 4),
                "sampling_cpu_sqa": round(samp_time, 4),
                "postprocess_greedy_cpu": round(pp_time, 4),
            },
            "problem_complexity": complexity,
            "qubo_parameters": {
                "mw_precision": self.mw_precision,
                "angle_precision": self.angle_precision,
                "penalty_balance": self.penalty_balance,
                "penalty_line": self.penalty_line,
                "seed": self.seed
            },
            "algorithmic_metrics": {
                "framework": "pure_qubo_discretization",
                "num_iterations": 1,
                "optimality_gap_percent": optimality_gap_percent,
                "sampler_stats": sampler_stats
            },
            "hardware_metrics": {
                "qpu_target": "classical_cpu_sqa_simulated",
                "logical_qubits": len(bqm.variables) if bqm else 0,
                "physical_qubits": len(bqm.variables) if bqm else 0,
                "max_chain_length": 1,
                "circuit_depth": 0
            }
        }

        return solution, metadata


if __name__ == "__main__":
    import copy
    import warnings

    import pandapower as pp
    import pandapower.networks as pn

    warnings.filterwarnings("ignore")
    net = pn.case5()

    net_ip = copy.deepcopy(net)
    pp.rundcopp(net_ip)
    ip_cost = float(net_ip.res_cost)

    # feasibility_tol_mw is pinned at 1.0 (a BaseQuboFormulator kwarg, forwarded through **kwargs)
    # so the feasibility bar stays fixed even if mw_precision is changed for experimentation.
    solver = SimulatedQuantumAnnealingSolver(
        formulation="dc_ptdf", num_reads=300, num_sweeps=5000, trotter_slices=32,
        mw_precision=1.0, feasibility_tol_mw=1.0, beta=8.0, gamma=1.0, seed=42,
    )
    solution, metadata = solver.solve(net, reference_cost_eur_per_hr=ip_cost)

    print(f"IP reference cost : {round(ip_cost, 3)} EUR/h")
    print(f"SQA cost          : {solution['cost_eur_per_hr']} EUR/h")
    print(f"Feasible          : {solution['grid_state']['feasibility']['is_feasible']}")
    print(f"Optimality gap    : {metadata['algorithmic_metrics']['optimality_gap_percent']} %")
    print(f"Sampler stats     : {metadata['algorithmic_metrics']['sampler_stats']}")
    print(f"Timings           : {metadata['execution_time_seconds']}")