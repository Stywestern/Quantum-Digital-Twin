"""Step 2: what happens between "embedded Ising model" and "what the annealer is actually programmed with".

  scale_to_hardware : rescale h, J to fill the device's programmable ranges (like auto_scale)
  quantise          : finite DAC resolution
  add_ice_noise     : Gaussian control errors on h and J (integrated control errors, ICE)

All of these work on flat numpy arrays (IsingArrays) for speed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import dimod
import numpy as np


@dataclass
class IsingArrays:
    labels: list          # physical qubit labels, order defines column order of samples
    h: np.ndarray         # (n,)
    rows: np.ndarray      # (m,) indices into labels
    cols: np.ndarray      # (m,)
    J: np.ndarray         # (m,)  one entry per coupler (u < v not required)

    @classmethod
    def from_bqm(cls, bqm: dimod.BinaryQuadraticModel) -> "IsingArrays":
        assert bqm.vartype is dimod.SPIN, "convert to SPIN first"
        labels = list(bqm.variables)
        idx = {v: i for i, v in enumerate(labels)}
        h = np.array([bqm.get_linear(v) for v in labels], dtype=float)
        quad = [(idx[u], idx[v], b) for u, v, b in bqm.iter_quadratic()]
        if quad:
            rows, cols, J = (np.array(x) for x in zip(*quad))
        else:
            rows = cols = np.zeros(0, dtype=int)
            J = np.zeros(0)
        return cls(labels, h, rows.astype(int), cols.astype(int), J.astype(float))

    def to_bqm(self) -> dimod.BinaryQuadraticModel:
        lin = {v: float(x) for v, x in zip(self.labels, self.h)}
        quad = {(self.labels[i], self.labels[j]): float(b)
                for i, j, b in zip(self.rows, self.cols, self.J)}
        return dimod.BinaryQuadraticModel.from_ising(lin, quad)

    def with_values(self, h=None, J=None) -> "IsingArrays":
        return replace(self, h=self.h if h is None else h, J=self.J if J is None else J)


def _limit(values: np.ndarray, lo: float, hi: float) -> float:
    f = np.inf
    if values.size:
        if values.max() > 0:
            f = min(f, hi / values.max())
        if values.min() < 0:
            f = min(f, lo / values.min())
    return f


def scale_to_hardware(ar: IsingArrays, h_range, j_range):
    """Multiply everything by one factor so that the tightest range is exactly filled.
    Ordering of energies is unchanged (only the overall scale). Returns (scaled, factor)."""
    f = min(_limit(ar.h, *h_range), _limit(ar.J, *j_range))
    if not np.isfinite(f):
        f = 1.0
    return ar.with_values(h=ar.h * f, J=ar.J * f), float(f)


def quantise(ar: IsingArrays, h_range, j_range, bits: int | None):
    """Round to a uniform grid with 2**bits levels across each range. bits=None -> no quantisation.
    Crude model of finite DAC precision (real devices are not exactly uniform)."""
    if bits is None:
        return ar
    def q(x, lo, hi):
        step = (hi - lo) / (2 ** bits)
        return np.round(x / step) * step
    return ar.with_values(h=q(ar.h, *h_range), J=q(ar.J, *j_range))


def add_ice_noise(ar: IsingArrays, sigma_h: float, sigma_j: float, rng: np.random.Generator):
    """Independent Gaussian offsets, in the *scaled* units (h, J of order 1). Applies to chain
    couplers too. sigma values are calibration parameters (step 5), not known constants."""
    h = ar.h + rng.normal(0.0, sigma_h, ar.h.shape) if sigma_h > 0 else ar.h
    J = ar.J + rng.normal(0.0, sigma_j, ar.J.shape) if sigma_j > 0 else ar.J
    return ar.with_values(h=h, J=J)


# =========================================================================
# Execution Block: Self-Test
# =========================================================================
if __name__ == "__main__":
    import pandapower.networks as nw
    
    try:
        from solvers.qubo_formulator import QuboFormulator
    except ImportError:
        print("[-] Could not import QuboFormulator. Ensure script is run from project root.")
        exit(1)

    print("=== Testing Analog Hardware Mapping Pipeline ===")
    
    # 1. Generate Logical BQM
    print("[1] Formulating case5 DC-PTDF QUBO...")
    net = nw.case5()
    if not net.ext_grid.empty:
        net.ext_grid['min_p_mw'] = 0.0
    
    # We use high precision (1 MW) so the dynamic range is naturally large
    formulator = QuboFormulator(formulation="dc_ptdf", mw_precision=1.0)
    bqm, _ = formulator._formulate_qubo(net)
    
    if len(bqm.variables) == 0:
        print("[-] ERROR: 0 variables. Domain Truncation deleted the BQM.")
        exit(1)
        
    print(f"    -> Logical BQM: {len(bqm.variables)} vars, {len(bqm.quadratic)} edges.")

    # 2. Abstract into IsingArrays
    print("\n[2] Converting to IsingArrays for bulk numpy operations...")
    ar_logical = IsingArrays.from_bqm(bqm.change_vartype(dimod.SPIN, inplace=False))
    max_h_orig = np.max(np.abs(ar_logical.h)) if len(ar_logical.h) > 0 else 0
    max_j_orig = np.max(np.abs(ar_logical.J)) if len(ar_logical.J) > 0 else 0
    print(f"    -> Original Max |h|: {max_h_orig:.2f}")
    print(f"    -> Original Max |J|: {max_j_orig:.2f}")
    
    # Check dynamic range (max / min_nonzero)
    nonzero_orig = [abs(v) for v in np.concatenate([ar_logical.h, ar_logical.J]) if abs(v) > 1e-10]
    dr_orig = max(nonzero_orig) / min(nonzero_orig) if nonzero_orig else 1.0
    print(f"    -> Original Dynamic Range: {dr_orig:,.1f}")

    # 3. Simulate Hardware Scaling (auto_scale)
    # Using typical Advantage values: h in [-4, 4], J in [-1, 1]
    print("\n[3] Scaling to hardware limits (h: [-4, 4], j: [-1, 1])...")
    h_range = (-4.0, 4.0)
    j_range = (-1.0, 1.0)
    
    ar_scaled, scale_factor = scale_to_hardware(ar_logical, h_range, j_range)
    print(f"    -> Scale Factor Applied: {scale_factor:e}")
    max_h_scaled = np.max(np.abs(ar_scaled.h)) if len(ar_scaled.h) > 0 else 0
    max_j_scaled = np.max(np.abs(ar_scaled.J)) if len(ar_scaled.J) > 0 else 0
    print(f"    -> Post-Scale Max |h|: {max_h_scaled:.4f}")
    print(f"    -> Post-Scale Max |J|: {max_j_scaled:.4f}")

    # 4. Simulate Finite DAC Precision (Quantization)
    print("\n[4] Simulating 5-bit DAC Quantization...")
    ar_quantized = quantise(ar_scaled, h_range, j_range, bits=5)
    
    # Calculate how many coefficients were erased to 0.0 by quantization
    nonzero_scaled_count = np.sum(np.abs(ar_scaled.J) > 1e-10)
    nonzero_quant_count = np.sum(np.abs(ar_quantized.J) > 1e-10)
    erased = nonzero_scaled_count - nonzero_quant_count
    print(f"    -> {erased} J-couplers (out of {len(ar_scaled.J)}) were erased to exactly 0.0 by DAC resolution.")

    # 5. Simulate Analog ICE Noise
    print("\n[5] Injecting Analog ICE Noise (sigma = 0.01)...")
    rng = np.random.default_rng(seed=42)
    # 0.01 is 1% of the full scale
    ar_noisy = add_ice_noise(ar_quantized, sigma_h=0.01, sigma_j=0.01, rng=rng)
    
    # Measure the RMS error introduced by noise and quantization combined
    err_h = np.sqrt(np.mean((ar_noisy.h - ar_scaled.h)**2))
    err_j = np.sqrt(np.mean((ar_noisy.J - ar_scaled.J)**2))
    print(f"    -> RMS |h| deviation from ideal: {err_h:.4f}")
    print(f"    -> RMS |J| deviation from ideal: {err_j:.4f}")
    
    print("\n[+] Analog mapping pipeline executed successfully.")