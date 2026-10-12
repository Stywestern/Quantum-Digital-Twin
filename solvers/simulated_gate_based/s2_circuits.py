"""QAOA circuits, two interchangeable backends producing the same final state:

    |psi(gamma, beta)> = prod_{l=p..1} [ exp(-i beta_l sum_i X_i)  exp(-i gamma_l H_C) ] |+>^n

  NumpyQAOA      exact statevector, cost layer applied as ONE elementwise phase exp(-i gamma E_norm[k]).
                 Fast: this is the backend for parameter training at n ~ 16-24 qubits.
  PennyLaneQAOA  gate-level circuit (RZ / IsingZZ / RX) on any PennyLane device. This is the "real"
                 circuit: use it for gate counts, noise studies (default.mixed), and to cross-validate
                 the numpy backend.

Parameter vector convention: params = [gamma_1..gamma_p, beta_1..beta_p]  (length 2p).
Mixer: exp(-i beta X) = RX(2 beta).
Cost: exp(-i gamma h Z) = RZ(2 gamma h), exp(-i gamma J ZZ) = IsingZZ(2 gamma J),
all with the NORMALISED coefficients h/scale, J/scale.
"""

from __future__ import annotations

import math
from collections import Counter
import numpy as np
import pennylane as qml
import networkx as nx

from solvers.simulated_gate_based.s1_ising import IsingProblem, Landscape


class QAOABase:
    def __init__(self, landscape: Landscape, p: int):
        self.land = landscape
        self.p = int(p)
        self.n_evals = 0

    def _split(self, params):
        params = np.asarray(params, dtype=float)
        if params.shape != (2 * self.p,):
            raise ValueError(f"expected {2 * self.p} parameters (p={self.p}), got shape {params.shape}")
        return params[:self.p], params[self.p:]

    def probs(self, params) -> np.ndarray:
        raise NotImplementedError

    def expectation(self, params) -> float:
        """<H_C> in normalised units (counts one circuit evaluation)."""
        self.n_evals += 1
        return float(self.probs(params) @ self.land.E_norm)


# ----------------------------------------------------------------------------------------------
# numpy exact backend
# ----------------------------------------------------------------------------------------------
def _rx_all(psi: np.ndarray, n: int, beta: float) -> None:
    """In place: apply exp(-i beta X) = cos(beta) I - i sin(beta) X to every qubit."""
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    for q in range(n):
        v = psi.reshape(1 << q, 2, 1 << (n - q - 1))
        a = v[:, 0, :].copy()
        v[:, 0, :] *= c
        v[:, 0, :] += s * v[:, 1, :]
        v[:, 1, :] *= c
        v[:, 1, :] += s * a


class NumpyQAOA(QAOABase):
    def __init__(self, landscape: Landscape, p: int):
        super().__init__(landscape, p)
        self._buf = np.empty(landscape.N, dtype=np.complex128)

    def probs(self, params) -> np.ndarray:
        gam, bet = self._split(params)
        n, N, E = self.land.n, self.land.N, self.land.E_norm
        psi = np.full(N, 1.0 / np.sqrt(N), dtype=np.complex128)
        for g, b in zip(gam, bet):
            np.multiply(E, -1j * g, out=self._buf)
            np.exp(self._buf, out=self._buf)
            psi *= self._buf
            _rx_all(psi, n, b)
        pr = psi.real ** 2 + psi.imag ** 2
        return pr / pr.sum()


# ----------------------------------------------------------------------------------------------
# PennyLane gate-level backend
# ----------------------------------------------------------------------------------------------
class PennyLaneQAOA(QAOABase):
    def __init__(self, landscape: Landscape, p: int, device: str = "default.mixed",
                 device_kwargs: dict | None = None, cost_layer: str = "gates",
                 hardware_preset: str = "ibm_eagle"):
        super().__init__(landscape, p)

        if device != "default.mixed":
            raise ValueError("Hardware emulation strictly requires device='default.mixed'")
        if cost_layer != "gates":
            raise ValueError("Hardware emulation strictly requires cost_layer='gates'")

        self.qml = qml
        self.cost_layer = cost_layer

        # Calculate routing overhead and hardware metrics
        self.hardware_metrics = resource_estimate(landscape.prob, p, target_hardware=hardware_preset)

        # IBM Eagle specifications (2023-2024 median calibrations)
        base_p1_error = 0.0003    # 0.03% single-qubit rotation error
        base_p2_error = 0.01      # 1.0% base CNOT error
        base_readout_err = 0.015  # 1.5% measurement readout bit-flip

        routing_multiplier = self.hardware_metrics["routing_overhead_multiplier"]

        self.noise = {
            "p1": min(0.99, base_p1_error),
            "p2": min(0.99, base_p2_error * routing_multiplier),
            "readout": min(0.99, base_readout_err)
        }

        prob: IsingProblem = landscape.prob
        n = prob.n
        self._h = prob.h / landscape.scale
        self._J = prob.J / landscape.scale
        self._rows, self._cols = prob.rows, prob.cols
        self.dev = qml.device(device, wires=n, **(device_kwargs or {}))

        def circuit(gammas, betas):
            # 1. Initial Superposition
            for q in range(n):
                qml.Hadamard(wires=q)

            # 2. QAOA p-Layers
            for l in range(self.p):
                self._apply_cost(gammas[l])
                for q in range(n):
                    qml.RX(2.0 * betas[l], wires=q)
                    qml.DepolarizingChannel(self.noise["p1"], wires=q)

            # 3. Measurement / Readout Error Layer
            for q in range(n):
                qml.BitFlip(self.noise["readout"], wires=q)

            return qml.probs(wires=list(range(n)))

        self._qnode = qml.QNode(circuit, self.dev)

    def _apply_cost(self, gamma):
        p1 = self.noise["p1"]
        p2 = self.noise["p2"]

        # Linear Terms (Magnetic Field)
        for i, hi in enumerate(self._h):
            if hi != 0.0:
                self.qml.RZ(2.0 * gamma * hi, wires=i)
                self.qml.DepolarizingChannel(p1, wires=i)

        # Quadratic Terms (Interactions)
        for a, b, jab in zip(self._rows, self._cols, self._J):
            a, b = int(a), int(b)
            self.qml.IsingZZ(2.0 * gamma * jab, wires=[a, b])
            self.qml.DepolarizingChannel(p2, wires=a)
            self.qml.DepolarizingChannel(p2, wires=b)

    def probs(self, params) -> np.ndarray:
        gam, bet = self._split(params)
        pr = np.asarray(self._qnode(gam, bet), dtype=float)
        return pr / pr.sum()


def make_circuit(landscape: Landscape, p: int, backend: str = "numpy", **kwargs) -> QAOABase:
    if backend == "numpy":
        return NumpyQAOA(landscape, p)
    if backend == "pennylane":
        return PennyLaneQAOA(landscape, p, **kwargs)
    raise ValueError("backend must be 'numpy' or 'pennylane'")


# ----------------------------------------------------------------------------------------------
# resource accounting (hardware-aware gate counts, depth, timing, and ESP)
# ----------------------------------------------------------------------------------------------
def resource_estimate(prob: IsingProblem, p: int, target_hardware: str = "ibm_eagle") -> dict:
    """Hardware-aware gate counts, depth, execution latency, and success probability."""
    n, m = prob.n, int(prob.J.size)
    n_h = int(np.count_nonzero(prob.h))

    # Base logical gate counts
    logical_cnot = 2 * m * p
    logical_1q = n + (n_h * p) + (n * p)  # Hadamards + RZ(h) + RX(beta)

    # Topological graph coloring & concurrency (Metric 4)
    colours = 0
    max_concurrent_cnot = 0
    avg_concurrent_cnot = 0.0

    if m:
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(zip(prob.rows.tolist(), prob.cols.tolist()))
        coloring = nx.greedy_color(nx.line_graph(g), strategy="largest_first")
        if coloring:
            colours = 1 + max(coloring.values())
            color_counts = Counter(coloring.values())
            max_concurrent_cnot = max(color_counts.values())
            avg_concurrent_cnot = round(m / colours, 2)

    if target_hardware in ["ibm_heavy_hex", "ibm_eagle"]:
        # Heavy-Hex routing: diameter scales as O(sqrt(n)).
        # Each SWAP adds 3 CNOTs and ~2 single-qubit correction gates.
        swap_distance = max(1.0, math.sqrt(n))
        physical_cnot = int(logical_cnot * swap_distance * 3)
        extra_swap_1q = int(logical_cnot * swap_distance * 2)
        physical_1q = logical_1q + extra_swap_1q
        routing_depth = int(math.sqrt(n) * 3)

        # Calibrated IBM Eagle hardware gate durations (ns)
        t_1q_ns = 35.0
        t_2q_ns = 300.0
        t_ro_ns = 750.0
        coherence_limit_ns = 100_000.0  # 100 µs nominal T2 limit

        # Base physical noise rates
        p1 = 0.0003
        p2_base = 0.01
        p_ro = 0.015
    else:
        # Ideal all-to-all logical baseline
        physical_cnot = logical_cnot
        physical_1q = logical_1q
        routing_depth = 1
        t_1q_ns = 0.0
        t_2q_ns = 0.0
        t_ro_ns = 0.0
        coherence_limit_ns = float("inf")
        p1, p2_base, p_ro = 0.0, 0.0, 0.0

    # Depth Breakdown
    depth_1q = 1 + (p if n_h else 0) + p
    depth_2q = p * (colours * routing_depth)
    depth_estimate = depth_1q + depth_2q

    # Metric 2: Duration & Coherence Saturation
    execution_duration_ns = (depth_1q * t_1q_ns) + (depth_2q * t_2q_ns) + t_ro_ns
    coherence_saturation_pct = (
        round((execution_duration_ns / coherence_limit_ns) * 100.0, 2)
        if coherence_limit_ns != float("inf") else 0.0
    )

    # Metric 1: Estimated Success Probability (ESP)
    esp = ((1.0 - p2_base) ** physical_cnot) * ((1.0 - p1) ** physical_1q) * ((1.0 - p_ro) ** n)

    routing_mult = round(physical_cnot / max(1, logical_cnot), 2)

    return {
        "qubits": n,
        "p": p,
        "logical_zz_terms": m,
        "logical_single_qubit_gates": logical_1q,
        "logical_two_qubit_gates": logical_cnot,
        "physical_single_qubit_gates": physical_1q,
        "physical_two_qubit_gates": physical_cnot,
        "routing_overhead_multiplier": routing_mult,
        "circuit_depth": depth_estimate,
        "depth_estimate": depth_estimate,
        "target_hardware": target_hardware,
        "concurrency": {
            "max_parallel_cnots": max_concurrent_cnot,
            "avg_parallel_cnots": avg_concurrent_cnot,
            "color_layers": colours
        },
        "timing": {
            "execution_duration_ns": round(execution_duration_ns, 1),
            "coherence_saturation_pct": coherence_saturation_pct
        },
        "fidelity": {
            "estimated_success_probability": float(esp)
        }
    }

if __name__ == "__main__":
    import numpy as np
    import time
    
    try:
        from solvers.qubo_formulator import QuboFormulator
        from homemade_grids.small_grids import case3_low_gen
    except ImportError:
        print("[-] Could not import grid/formulator. Ensure script is run from project root.")
        exit(1)

    print("=== Gate-Based Pipeline: Ideal Baseline vs. Noisy IBM Emulation ===")
    
    # 1. Load the 3-Bus Grid
    net = case3_low_gen()
    if hasattr(net, 'ext_grid') and not net.ext_grid.empty:
        net.ext_grid['min_p_mw'] = 0.0
        
    formulator = QuboFormulator(formulation="dc_ptdf", mw_precision=50.0)
    bqm, _ = formulator._formulate_qubo(net)
    
    prob = IsingProblem.from_bqm(bqm)
    landscape = Landscape(prob, normalize=True)
    
    print(f"[1] Problem Loaded: {prob.n} Logical Qubits.")
    if prob.n > 12:
        print("    [FATAL] Exceeds 12 qubits. Density matrix simulation (2^(2n)) "
              "will exceed memory limits. Aborting.")
        exit(1)

    # 2. Hardware Resource Estimation
    p_layers = 1
    ibm_metrics = resource_estimate(prob, p=p_layers, target_hardware="ibm_heavy_hex")
    
    print("\n[2] IBM Heavy-Hex Resource Estimation (p=1 layer):")
    print(f"    -> Logical 2-Qubit Interactions : {ibm_metrics['logical_two_qubit_gates'] // 2}")
    print(f"    -> Physical CNOT Gates (SWAP Routed): {ibm_metrics['physical_two_qubit_gates']}")
    print(f"    -> Routing Overhead Multiplier      : {ibm_metrics['routing_overhead_multiplier']:.2f}x")
    print(f"    -> Circuit Depth Estimate          : {ibm_metrics['depth_estimate']} layers")

    # 3. Initialize the Two Core Backends
    print("\n[3] Initializing QAOA Backends...")
    # Fast, exact statevector simulator (Ideal mathematical baseline)
    qaoa_numpy = make_circuit(landscape, p=p_layers, backend="numpy")
    
    # Noisy gate-level density matrix simulator (IBM Eagle physical emulator)
    qaoa_ibm = make_circuit(landscape, p=p_layers, backend="pennylane", hardware_preset="ibm_eagle")