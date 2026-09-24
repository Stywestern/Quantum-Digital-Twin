import math
import time
from dataclasses import dataclass, field

import numpy as np
import dimod
import dwave_networkx as dnx
import minorminer
import dwave.embedding as de
from dwave.embedding.chain_strength import uniform_torque_compensation

from solvers.qubo_formulator import QuboFormulator

try:
    import openjij as oj
except ImportError:
    raise ImportError("OpenJij is required for SQA. Run: pip install openjij")

try:
    from dwave.samplers import SteepestDescentSolver
    _HAVE_GREEDY = True
except ImportError:
    _HAVE_GREEDY = False

# ---------------------------------------------------------------------- #
# Device profile: every hardware constant below is either read from D-Wave's
# published documentation (cited inline) or an EXPLICITLY FLAGGED assumption
# to be replaced with a live-queried value once cloud access exists. None of
# these numbers were fitted to make this problem solve well -- that is the
# whole point of separating them from the solver's own tunables.
# ---------------------------------------------------------------------- #
@dataclass
class DeviceProfile:
    """
    A specific real QPU's documented operating envelope. Defaults describe a
    modern Advantage-class QPU as of D-Wave's public docs (docs.dwavequantum.com,
    docs.dwavesys.com; checked at the time this file was written). Two kinds of
    field:
      - DOCUMENTED: taken directly from D-Wave's published solver properties /
        technical documentation. Still worth confirming against the specific
        solver you will actually submit to (`DWaveSampler().properties`), since
        these vary by QPU generation and even by individual chip.
      - ASSUMED: no single authoritative number exists publicly (D-Wave's own
        docs state there is "not a simple number for how precise h and J values
        can be set" -- ICE depends on the local neighbourhood of each qubit, not
        a flat spec). These are placeholders to be REPLACED by an empirical
        measurement once you have cloud access -- see the class docstring of
        SimulatedQuantumAnnealingSolver for how.
    """
    name: str = "advantage_pegasus_p16_generic"
    topology: str = "pegasus"          # "pegasus" or "zephyr"
    topology_size: int = 16            # Pegasus P16 (~5612 active qubits) or Zephyr Z12 (~4516)
    max_degree: int = 15               # DOCUMENTED: Pegasus=15, Zephyr=20 (docs.dwavequantum.com)
    h_range: tuple = (-4.0, 4.0)       # DOCUMENTED (nominal, current-generation Advantage; older/other
                                        # solvers may report [-2, 2] -- confirm via sampler.properties)
    j_range: tuple = (-1.0, 1.0)       # DOCUMENTED (dwave.system.DWaveSampler().properties['j_range'])
    temperature_mK: float = 15.0       # DOCUMENTED order of magnitude ("generally operates below 20 mK");
                                        # 15 mK is a representative point in that range, not a specific
                                        # solver's calibrated value
    annealing_time_range_us: tuple = (0.5, 2000.0)   # DOCUMENTED for Advantage_system4.1-class solvers;
                                        # older solvers had a 1.0 us floor -- confirm per solver
    programming_time_us: float = 12000.0   # DOCUMENTED order of magnitude (~5500-14200 us reported across
                                        # solvers in the literature); READ THE REAL VALUE from
                                        # sampler.properties['problem_timing_data'] for your solver --
                                        # this materially changes any wall-clock/TTS accounting, since it
                                        # is fixed PER PROBLEM SUBMISSION, independent of anneal time
    assumed_ice_relative_precision: float = 0.01   # ASSUMED, NOT DOCUMENTED. Placeholder: a coefficient
                                        # whose magnitude, after auto_scale, is below this fraction of the
                                        # device's h/J range is flagged "at risk" of being lost to
                                        # integrated control error. There is no published single number for
                                        # this -- replace it with an empirically measured value (e.g. from
                                        # repeated small-problem submissions comparing intended vs. realized
                                        # h/J) once you have cloud access. Until then, treat any
                                        # feasibility verdict driven by this number as provisional.

    def target_graph(self):
        if self.topology == "zephyr":
            return dnx.zephyr_graph(self.topology_size)
        return dnx.pegasus_graph(self.topology_size)


# A second, older-generation profile for comparison / sensitivity checks (values per the same sources).
ADVANTAGE2_ZEPHYR_PROFILE = DeviceProfile(
    name="advantage2_zephyr_z12_generic", topology="zephyr", topology_size=12, max_degree=20,
    h_range=(-4.0, 4.0), j_range=(-1.0, 1.0), temperature_mK=15.0,
    annealing_time_range_us=(0.5, 2000.0), programming_time_us=5500.0,
)


class SimulatedQuantumAnnealingSolver(QuboFormulator):
    """
    SQA (path-integral Monte Carlo on the transverse-field Ising model, via OpenJij's SQASampler)
    applied to the DC-OPF QUBO built by BaseQuboFormulator, built to be AS CLOSE AS REASONABLY
    POSSIBLE to what would actually happen if this exact problem were submitted to a real D-Wave
    QPU -- so that a later cloud run against the SAME embedded problem is an actual test of how
    good this simulation's assumptions were, not a different problem in disguise.

    What "as accurate as possible" means here, concretely, and what it still can't capture
    -------------------------------------------------------------------------------------
    Modelled (this file):
      1. HARDWARE COEFFICIENT RANGE. h, J are auto_scale'd to the DEVICE's documented h_range/
         j_range (see `_hardware_feasibility_report`), exactly as D-Wave's own auto_scale does
         before a real submission -- not to an arbitrary "max coefficient = 1", which was this
         file's earlier (incorrect) normalization.
      2. A NAMED HARDWARE GRAPH. `minorminer.find_embedding` runs against an actual Pegasus/Zephyr
         graph from `dwave_networkx`, seeded for reproducibility. `dwave.embedding.embed_bqm` then
         builds the ACTUAL embedded BQM -- chains and all -- using `chain_strength.
         uniform_torque_compensation`, D-Wave's own recommended heuristic (not one invented here).
         THIS embedded BQM, not the logical one, is what gets sampled below.
      3. CHAIN BREAKS, insofar as classical SQA has any notion of them at all: `unembed_sampleset`
         with `chain_break_method=majority_vote` decodes chains back to logical variables and
         reports the resulting `chain_break_fraction`, exactly as you would inspect it on a real
         QPU response.
      4. A fixed, documented device temperature and anneal-time range (`DeviceProfile`), so no
         per-run tuning of "how quantum" the run is (see `_beta_from_temperature`).

    NOT modelled, and this is a genuine limit of classical simulation, not an oversight to code
    around (see the earlier discussion in this conversation for why):
      - SQA is a Monte Carlo sampler of thermal equilibrium in imaginary time. It is not a
        simulation of the real annealer's non-equilibrium, continuous-time, open-quantum-system
        dynamics. Chain-break statistics reported here reflect classical SQA's OWN dynamics on the
        embedded graph, not a measurement of what a physical chain does under real flux/thermal
        noise -- treat `chain_break_fraction` here as a lower-effort proxy, not a prediction.
      - The `assumed_ice_relative_precision` used in the feasibility report is EXPLICITLY a
        placeholder (see DeviceProfile docstring) -- D-Wave does not publish a single ICE number,
        because it depends on each qubit's local coupling neighbourhood, not a flat spec.
      - Non-stoquastic effects, leakage, 1/f flux noise spectra, and anneal-schedule filtering
        (the QPU's own low-pass filter on h/J waveforms) are not modelled at all.

    The point of building it this way: once you have cloud access, submit the SAME
    `embedded_bqm`/`embedding` this file produces (via `FixedEmbeddingComposite(DWaveSampler(),
    embedding)`, see `solve()`'s returned `reproducibility` block) to the real QPU, and compare.
    The gap between that result and this one is your actual, empirically measured answer to "how
    much do these unmodelled effects matter for this problem class" -- which is the only way to
    close this loop, as discussed earlier. This file cannot get you to certainty on its own; it is
    built so that when hardware access exists, the comparison it enables is a clean one.

    solve() vs. characterize(): see the previous version's docstring for why `characterize()` (not
    included in this hardware-aware version yet -- ask for it if you want anneal-time-vs-quality
    curves through the SAME embedding-aware pipeline) is the right tool for resource-vs-quality
    claims, and why a single `solve()` only tells you "did it happen to work this time."
    """

    def __init__(self, formulation="dc_ptdf", num_reads=300, num_sweeps=1000, max_time=1800,
                 mw_precision=1.0, trotter_slices=32, device=None, postprocess_greedy=True,
                 embedding_seed=None, seed=None, **kwargs):
        """
        device            : a DeviceProfile (default: a generic modern Pegasus-class Advantage
                            profile). Every hardware constant lives here, not as a free per-run
                            solver parameter -- see the class docstring.
        postprocess_greedy: applies dwave-samplers' SteepestDescentSolver to every read, exactly as
                            is common practice on real QPU output. Both the raw and the polished
                            result are always kept separate in `sampler_stats` (`energy_best_raw`
                            vs. `energy_best_after_greedy`) -- ANY claim drawn from this solver's
                            output should say explicitly which one it is using.
        embedding_seed    : seeds `minorminer.find_embedding` for a reproducible embedding across
                            calls (minorminer's search is otherwise stochastic).
        seed              : seeds a numpy Generator that draws one independent per-read seed for
                            each of the `num_reads` SQA calls -- see the extensive earlier notes in
                            this conversation on why this loop (not a single batched, seeded call)
                            is the reproducible-AND-diverse approach in the installed OpenJij
                            version.
        Any BaseQuboFormulator kwarg (feasibility_tol_mw, penalty_balance, penalty_line, encoding,
        ...) can be passed through.
        """
        super().__init__(
            formulation=formulation, max_time=max_time, mw_precision=mw_precision, **kwargs
        )

        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.trotter_slices = trotter_slices
        self.device = device or DeviceProfile()
        self.postprocess_greedy = postprocess_greedy and _HAVE_GREEDY
        self.embedding_seed = embedding_seed
        self.seed = seed

        self.sampler = oj.SQASampler()

    # ------------------------------------------------------------------ #
    # 1. Hardware coefficient-range feasibility (replaces the old arbitrary normalization)
    # ------------------------------------------------------------------ #
    def _hardware_feasibility_report(self, bqm):
        """
        Mirrors D-Wave's own `auto_scale`: h, J are divided by whatever factor is needed (>= 1, i.e.
        only ever scaled DOWN) to fit within `self.device.h_range` / `j_range`, then checked against
        `assumed_ice_relative_precision` (see DeviceProfile docstring for why that number is a
        placeholder, not a fact). Returns the scaled (h, J, offset), the scale factor, and which
        terms are "at risk" of being lost to control error at this device's coefficient range.
        """
        h, J, offset = bqm.to_ising()
        h_max_dev, j_max_dev = self.device.h_range[1], self.device.j_range[1]

        max_h = max((abs(v) for v in h.values()), default=0.0)
        max_j = max((abs(v) for v in J.values()), default=0.0)
        required_scale = max(1.0, max_h / h_max_dev if h_max_dev > 0 else 0.0,
                              max_j / j_max_dev if j_max_dev > 0 else 0.0)

        h_scaled = {k: v / required_scale for k, v in h.items()}
        J_scaled = {k: v / required_scale for k, v in J.items()}

        h_floor = self.device.assumed_ice_relative_precision * h_max_dev
        j_floor = self.device.assumed_ice_relative_precision * j_max_dev
        at_risk_h = [k for k, v in h_scaled.items() if 0 < abs(v) < h_floor]
        at_risk_j = [k for k, v in J_scaled.items() if 0 < abs(v) < j_floor]
        total_terms = len(h_scaled) + len(J_scaled)
        at_risk_fraction = (len(at_risk_h) + len(at_risk_j)) / total_terms if total_terms else 0.0

        nonzero = [abs(v) for v in list(h_scaled.values()) + list(J_scaled.values()) if abs(v) > 1e-15]
        report = {
            "required_auto_scale_factor": round(required_scale, 6),
            "post_scale_max_abs_h": round(max(abs(v) for v in h_scaled.values()), 6) if h_scaled else 0.0,
            "post_scale_max_abs_j": round(max(abs(v) for v in J_scaled.values()), 6) if J_scaled else 0.0,
            "post_scale_min_nonzero_abs": round(min(nonzero), 8) if nonzero else 0.0,
            "post_scale_dynamic_range": round(max(nonzero) / min(nonzero), 2) if nonzero else 1.0,
            "assumed_ice_relative_precision": self.device.assumed_ice_relative_precision,
            "at_risk_term_count": len(at_risk_h) + len(at_risk_j),
            "total_term_count": total_terms,
            "at_risk_fraction": round(at_risk_fraction, 4),
            "verdict": (
                "UNRELIABLE ON REAL HARDWARE (under current placeholder ICE assumption): "
                f"{round(100*at_risk_fraction, 1)}% of coefficients are smaller than the assumed "
                "control-error floor and may be realized as zero or noise-dominated on a real QPU."
                if at_risk_fraction > 0.05 else
                "Plausible fit within the device's coefficient range under the current placeholder "
                "ICE assumption -- still unverified against a real measurement."
            ),
        }
        return h_scaled, J_scaled, offset, report

    # ------------------------------------------------------------------ #
    # 2. Embedding onto a named hardware graph, and the actual embedded BQM
    # ------------------------------------------------------------------ #
    def _embed(self, h_scaled, J_scaled, offset):
        ising_bqm = dimod.BinaryQuadraticModel.from_ising(h_scaled, J_scaled, offset)
        target_graph = self.device.target_graph()
        source_edgelist = list(ising_bqm.quadratic.keys())

        mm_kwargs = {} if self.embedding_seed is None else {"random_seed": self.embedding_seed}
        embedding = minorminer.find_embedding(source_edgelist, list(target_graph.edges()), **mm_kwargs)
        if not embedding:
            raise RuntimeError(
                f"Minor-embedding failed onto {self.device.name}: the logical problem "
                f"({len(ising_bqm.variables)} variables, {len(source_edgelist)} quadratic terms) did "
                f"not fit this device's graph (max degree {self.device.max_degree})."
            )

        chain_str = uniform_torque_compensation(ising_bqm, embedding)
        embedded_bqm = de.embed_bqm(ising_bqm, embedding, target_graph.adj, chain_strength=chain_str)

        physical_qubits = sum(len(chain) for chain in embedding.values())
        max_chain_length = max(len(chain) for chain in embedding.values())
        embed_report = {
            "device": self.device.name,
            "logical_qubits": len(ising_bqm.variables),
            "logical_interactions": len(source_edgelist),
            "physical_qubits": physical_qubits,
            "max_chain_length": max_chain_length,
            "mean_chain_length": round(physical_qubits / len(embedding), 3) if embedding else 0.0,
            "chain_strength": round(float(chain_str), 6),
            "embedding_seed": self.embedding_seed,
        }
        return embedded_bqm, embedding, ising_bqm, embed_report

    # ------------------------------------------------------------------ #
    def solve_opf(self, net, reference_cost_eur_per_hr=None):
        start_time = time.time()
        status, cost, dispatch, sgen_dispatch, slack_dispatch, feasibility = "Success", None, {}, {}, {}, {}
        form_time, embed_time, samp_time, unembed_time, pp_time = 0.0, 0.0, 0.0, 0.0, 0.0
        
        bqm, complexity = None, {}
        sampler_stats, hw_feasibility_report, embed_report = {}, {}, {}

        try:
            # 1. Formulation
            t0 = time.time()
            bqm, complexity = self._formulate_qubo(net)
            form_time = time.time() - t0

            # 2. Hardware coefficient-range check + device auto_scale (NOT arbitrary normalization)
            h_scaled, J_scaled, ising_offset, hw_feasibility_report = self._hardware_feasibility_report(bqm)

            # 3. Real minor-embedding onto the device's actual graph, real chain strength, real embedded BQM -- this, not the logical bqm, is what gets sampled.
            t1 = time.time()
            embedded_bqm, embedding, ising_bqm, embed_report = self._embed(h_scaled, J_scaled, ising_offset)
            embed_time = time.time() - t1

            # 4. Sample the EMBEDDED problem
            t2 = time.time()
            response = self.sampler.sample(
                embedded_bqm, 
                trotter=self.trotter_slices, 
                num_sweeps=self.num_sweeps,
                num_reads=self.num_reads, 
                seed=self.seed
            )
            samp_time = time.time() - t2

            # Unembed the bulk sampleset all at once
            t_unembed = time.time()
            unembedded = de.unembed_sampleset(
                response, embedding, ising_bqm,
                chain_break_method=de.majority_vote, chain_break_fraction=True
            )
            unembed_time = time.time() - t_unembed

            logical_samples = list(unembedded.samples())
            raw_energies = unembedded.record.energy.astype(float)
            
            if hasattr(unembedded.record, 'chain_break_fraction'):
                chain_break_fracs = unembedded.record.chain_break_fraction.astype(float)
            else:
                chain_break_fracs = np.zeros(len(raw_energies))

            raw_energies = np.array(raw_energies, dtype=float)
            best_idx = int(np.argmin(raw_energies))
            best_raw_sample = logical_samples[best_idx]     # Ising {-1,+1}; converted to BINARY below
            best_raw_energy = float(raw_energies[best_idx])
            mean_chain_break = float(np.mean(chain_break_fracs))

            # Ising -> BINARY, since _decode_solution / the QUBO objective are all defined in BINARY.
            def to_binary_sample(spin_sample):
                return {k: (1 if v == 1 else 0) for k, v in spin_sample.items()}

            best_raw_sample_binary = to_binary_sample(best_raw_sample)

            # 5. Optional greedy post-processing, on the ORIGINAL LOGICAL bqm (not the embedded one, once unembedded, chains no longer exist as a concept). Raw and polished are BOTH kept.
            best_sample, best_energy, polish_gain = best_raw_sample_binary, best_raw_energy, 0.0
            if self.postprocess_greedy:
                t3 = time.time()
                all_binary_samples = [to_binary_sample(s) for s in logical_samples]
                greedy = SteepestDescentSolver().sample(bqm, initial_states=all_binary_samples)
                pp_time = time.time() - t3
                g_best = greedy.first
                if g_best.energy < best_raw_energy:
                    best_sample, best_energy = g_best.sample, float(g_best.energy)
                    polish_gain = best_raw_energy - best_energy

            sampler_stats = {
                "num_reads_requested": self.num_reads,
                "num_sweeps": self.num_sweeps,
                "trotter_slices": self.trotter_slices,
                "mean_chain_break_fraction": round(mean_chain_break, 4),
                "chain_break_fraction_note": (
                    "From classical SQA's OWN dynamics on the embedded graph, not a measurement of a "
                    "real chain under physical noise -- see class docstring."
                ),
                "energy_best_raw": round(best_raw_energy, 2),
                "energy_mean_raw": round(float(raw_energies.mean()), 2),
                "energy_worst_raw": round(float(raw_energies.max()), 2),
                "energy_std_dev_raw": round(float(raw_energies.std()), 2),
                "greedy_postprocess_applied": self.postprocess_greedy,
                "greedy_polish_gain_eur": round(float(polish_gain), 2),
                "energy_best_after_greedy": round(float(best_energy), 2),
            }

            # 6. Decoding, via BaseQuboFormulator, from the LOGICAL sample (unembedded already).
            dispatch, sgen_dispatch, slack_dispatch, cost, feasibility = self._decode_solution(best_sample, net)

        except Exception as e:
            status = f"Failed: {str(e)}"
            print(f"\nCRITICAL ERROR: {e}\n")
            raise

        exec_time = round(time.time() - start_time, 4)

        optimality_gap_percent = None
        if reference_cost_eur_per_hr and feasibility.get("is_feasible"):
            optimality_gap_percent = round(
                100.0 * (cost - reference_cost_eur_per_hr) / reference_cost_eur_per_hr, 4)

        # Calculate the max line loading percentage dynamically
        max_load_pct = None
        if hasattr(self, '_lines') and self._lines:
            loadings = []
            for ln in self._lines:
                if ln['p_max'] > 0.0:
                    flow = abs(feasibility["line_flows_mw"].get(ln['idx'], 0.0))
                    loadings.append((flow / ln['p_max']) * 100.0)
            if loadings:
                max_load_pct = round(max(loadings), 3)

        solution = {
            "cost_eur_per_hr": cost,
            "generator_dispatch_mw": dispatch,
            "static_generator_dispatch_mw": sgen_dispatch,
            "slack_dispatch_mw": slack_dispatch,
            "grid_state": {"max_line_loading_percent": max_load_pct, "feasibility": feasibility},
        }

        # `reproducibility`: everything needed to submit THIS SAME embedded problem to a real QPU
        # later via FixedEmbeddingComposite(DWaveSampler(), embedding), for a like-for-like check against this simulation's assumptions.
        reproducibility = {
            "embedding": {int(k) if isinstance(k, (int, np.integer)) else k: list(v) for k, v in embedding.items()} if embedding else None,
            "device_profile": self.device.name,
            "chain_strength": embed_report.get("chain_strength"),
        }

        metadata = {
            "status": status,
            "solver_name": f"openjij_sqa_hardware_aware_{self.formulation}",
            "execution_time_seconds": {
                "total": exec_time, 
                "formulation_cpu": round(form_time, 4),
                "embedding_cpu": round(embed_time, 4), 
                "sampling_cpu_sqa": round(samp_time, 4),
                "unembedding_cpu": round(unembed_time, 4),
                "postprocess_greedy_cpu": round(pp_time, 4),
            },
            "problem_complexity": complexity,
            "hardware_feasibility_report": hw_feasibility_report,
            "hardware_metrics": {
                "qpu_target": self.device.name,
                "logical_qubits": embed_report.get("logical_qubits", 0),
                "physical_qubits": embed_report.get("physical_qubits", 0),
                "max_chain_length": embed_report.get("max_chain_length", 0),
                "mean_chain_length": embed_report.get("mean_chain_length", 0.0),
                "chain_strength": embed_report.get("chain_strength", 0.0),
                "device_temperature_mK": self.device.temperature_mK,
                "device_annealing_time_range_us": self.device.annealing_time_range_us,
                "device_programming_time_us": self.device.programming_time_us,
            },
            "algorithmic_metrics": {
                "framework": "hardware_aware_sqa",
                "optimality_gap_percent": optimality_gap_percent,
                "sampler_stats": sampler_stats,
            },
            "reproducibility": reproducibility,
        }
        return solution, metadata

    def solve_pf(self, net):
        """
        Executes a QUBO Power Flow by locking economic variables to their current setpoints.
        Used strictly to validate physical grid state (Kirchhoff's laws) devoid of economics.
        """
        import copy
        net_pf = copy.deepcopy(net)
        
        # 1. Lock dispatchable assets to eliminate decision variables
        for idx in net_pf.gen.index:
            p = float(net_pf.gen.at[idx, 'p_mw']) if not np.isnan(net_pf.gen.at[idx, 'p_mw']) else 0.0
            net_pf.gen.at[idx, 'min_p_mw'] = p
            net_pf.gen.at[idx, 'max_p_mw'] = p
            
        for idx in net_pf.ext_grid.index:
            if 'p_mw' in net_pf.ext_grid.columns and not np.isnan(net_pf.ext_grid.at[idx, 'p_mw']):
                p = float(net_pf.ext_grid.at[idx, 'p_mw'])
            else:
                p = 0.0
            net_pf.ext_grid.at[idx, 'min_p_mw'] = p
            net_pf.ext_grid.at[idx, 'max_p_mw'] = p

        # 2. Strip cost polynomials so the BQM energy is purely physical mismatch
        net_pf.poly_cost = net_pf.poly_cost.iloc[0:0]
        if hasattr(net_pf, 'pwl_cost'):
            net_pf.pwl_cost = net_pf.pwl_cost.iloc[0:0]

        # 3. Solve using the core SQA workflow
        # (reference_cost is irrelevant for PF)
        solution, metadata = self.solve_opf(net_pf, reference_cost_eur_per_hr=None)
        
        # 4. Tweak outputs for PF context
        solution["cost_eur_per_hr"] = None
        metadata["solver_name"] = f"openjij_sqa_pf_{self.formulation}"
        metadata["algorithmic_metrics"]["framework"] = "hardware_aware_sqa_pf"
        
        return solution, metadata


if __name__ == "__main__":
    import copy
    import warnings

    import pandapower as pp
    import pandapower.networks as pn

    from homemade_grids.small_grids import case3_low_gen

    warnings.filterwarnings("ignore")
    net = case3_low_gen()

    net_ip = copy.deepcopy(net)
    pp.rundcopp(net_ip)
    ip_cost = float(net_ip.res_cost)

    solver = SimulatedQuantumAnnealingSolver(
        formulation="dc_ptdf", mw_precision=1.0, feasibility_tol_mw=1.0,
        trotter_slices=16, num_reads=20, num_sweeps=500,
        embedding_seed=42, seed=42,
    )
    solution, metadata = solver.solve_opf(net, reference_cost_eur_per_hr=ip_cost)

    print("=== Hardware feasibility report (auto_scale + placeholder ICE check) ===")
    for k, v in metadata["hardware_feasibility_report"].items():
        print(f"  {k}: {v}")

    print("\n=== Embedding / hardware metrics ===")
    for k, v in metadata["hardware_metrics"].items():
        print(f"  {k}: {v}")

    print("\n=== Solution ===")
    print(f"IP reference cost : {round(ip_cost, 3)} EUR/h")
    print(f"SQA cost          : {solution['cost_eur_per_hr']} EUR/h")
    print(f"Feasible          : {solution['grid_state']['feasibility']['is_feasible']}")
    print(f"Optimality gap    : {metadata['algorithmic_metrics']['optimality_gap_percent']} %")
    print(f"Sampler stats     : {metadata['algorithmic_metrics']['sampler_stats']}")
    print(f"Timings           : {metadata['execution_time_seconds']}")