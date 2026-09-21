import heapq
import math

import dimod
import numpy as np

from solvers.base_solver import BaseSolver

class BaseQuboFormulator(BaseSolver):
    """
    Translates Pandapower DC-OPF problems into pure QUBO/Ising models.
    Does not execute solvers. Designed to be inherited by SA, QA, and QAOA solvers.

    Public interface for subclasses (unchanged):
        _formulate_qubo(net)          -> (bqm, complexity)
        _decode_solution(sample, net) -> (gen_dispatch, sgen_dispatch, ext_dispatch, cost, feasibility)
        view_problem_definition(net)

    Discretisation summary
    ----------------------
    * Every bounded continuous variable x in [lo, hi] is written as
      x = lo + sum_k w_k * b_k with a *bounded-coefficient* binary expansion whose
      weights sum to exactly (hi - lo). The encoding can therefore never exceed hi.
    * Bus angles get a per-bus range (from line ratings, see _compute_angle_bounds) and
      a per-bus resolution chosen so that one angle step moves a line flow by at most
      `mw_precision` MW. That keeps the KCL equalities satisfiable on the MW grid and
      keeps QUBO coefficients on the MW scale.
    * penalty_balance / penalty_line default to values derived from the cost scale
      (see _resolve_penalties) so that violating a constraint never pays off.
    """

    # Line ratings that are missing / absurdly large are treated as "unconstrained".
    UNCONSTRAINED_I_KA = 100.0
    UNCONSTRAINED_MW = 5000.0
    # Dispatch upper bound used when an asset has no max_p_mw.
    DEFAULT_MAX_MW = 500.0
    # Angle ranges are widened by this factor so penalised (slightly violating) states stay representable.
    ANGLE_HEADROOM = 1.1

    _DISPATCH_TYPES = ("gen", "sgen", "ext_grid")
    _BIT_PREFIX = {"gen": "gen", "sgen": "sgen", "ext_grid": "ext"}

    def __init__(self, formulation="dc", max_time=1800, mw_precision=1.0,
                 angle_precision=None, penalty_balance=None, penalty_line=None,
                 penalty_safety=2.0, feasibility_tol_mw=None,
                 enforce_line_limits=True, fallback_angle_rad=1.0, **kwargs):
        """
        angle_precision     : None -> derived per bus as mw_precision / (largest susceptance at that bus).
                              A float forces one uniform resolution (rad) for all buses.
        penalty_balance/line: None -> derived from the cost scale: penalty_safety * max_marginal_cost / mw_precision.
        feasibility_tol_mw  : tolerance used by the feasibility check; None -> mw_precision.
        fallback_angle_rad  : half-range for angles of buses that have no path to a slack bus.
        """
        super().__init__(max_time=max_time) if hasattr(super(), '__init__') else None

        if formulation != "dc":
            raise NotImplementedError("Only the 'dc' formulation is implemented.")

        self.formulation = formulation
        self.mw_precision = mw_precision
        self.angle_precision = angle_precision
        self.penalty_balance = penalty_balance
        self.penalty_line = penalty_line
        self.penalty_safety = penalty_safety
        self.feasibility_tol_mw = feasibility_tol_mw
        self.enforce_line_limits = enforce_line_limits
        self.fallback_angle_rad = fallback_angle_rad

        self.var_registry = {}
        self.cost_model_warnings = []
        self.lambda_balance = penalty_balance
        self.lambda_line = penalty_line

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _val(x, default):
        """float(x), or `default` when x is None / NaN / not convertible (pandas returns NaN, not the .get default)."""
        try:
            if x is None or math.isnan(float(x)):
                return default
            return float(x)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _active(net, et):
        tbl = getattr(net, et, None)
        if tbl is None or tbl.empty:
            return []
        return list(tbl.index[tbl.in_service.astype(bool)])

    def _asset_bounds(self, net, et, idx):
        row = getattr(net, et).loc[idx]
        p_min = self._val(row.get('min_p_mw'), 0.0)
        default_max = self._val(row.get('p_mw'), 0.0) if et == 'sgen' else self.DEFAULT_MAX_MW
        p_max = self._val(row.get('max_p_mw'), default_max)
        if p_max < p_min - 1e-9:
            raise ValueError(f"{et} {idx}: max_p_mw ({p_max}) < min_p_mw ({p_min}); check the limits/sign convention.")
        return p_min, p_max

    def _get_radix_weights(self, total_range, precision):
        """
        Bounded-coefficient binary weights whose sum is exactly `total_range` (in physical units).
        Weights are precision*{1,2,4,...,2^(K-1)} plus one remainder weight; all values in
        [0, total_range] are reachable with gaps smaller than `precision`, and nothing above total_range is.
        """
        if total_range <= 1e-12:
            return []
        n = max(1, int(math.floor(total_range / precision + 1e-9)))
        K = int(math.floor(math.log2(n)))
        weights = [precision * (2 ** k) for k in range(K)]
        weights.append(total_range - (2 ** K - 1) * precision)
        return weights

    def _register(self, group, key, prefix, lo, hi, precision):
        weights = self._get_radix_weights(hi - lo, precision) if hi > lo else []
        bits = [f"{prefix}_bit_{k}" for k in range(len(weights))]
        self.var_registry[group][key] = {'min': lo, 'max': hi, 'precision': precision,
                                         'bits': bits, 'weights': weights}
        return len(bits)

    @staticmethod
    def _decode_value(reg, sample):
        return reg['min'] + sum(w for bit, w in zip(reg['bits'], reg['weights']) if sample.get(bit, 0) == 1)

    # ------------------------------------------------------------------ #
    # Cost model (single source of truth for objective, decode and printing)
    # ------------------------------------------------------------------ #
    def _fit_pwl_quadratic(self, segments, p_lo, p_hi, label):
        """
        pandapower pwl_cost points are segments (p_from, p_to, slope_eur_per_mw).
        A QUBO objective can only hold polynomial terms, so the PWL curve is replaced by its
        least-squares quadratic over the asset's dispatch range (exact for a single segment).
        """
        segs = []
        for seg in segments:
            if len(seg) != 3:
                raise ValueError(f"pwl_cost of {label}: expected segments (p_from, p_to, slope), got {seg!r}")
            segs.append((float(seg[0]), float(seg[1]), float(seg[2])))
        segs.sort()

        def pwl(p):
            total = np.zeros_like(p, dtype=float)
            for k, (a, b, slope) in enumerate(segs):
                if k == len(segs) - 1:
                    total += slope * np.maximum(p - a, 0.0)          # last slope extends upwards
                else:
                    total += slope * np.clip(p - a, 0.0, b - a)
            return total

        if p_hi <= p_lo:
            return (float(pwl(np.array([p_lo]))[0]), 0.0, 0.0)

        xs = np.linspace(p_lo, p_hi, 201)
        ys = pwl(xs)
        c2, c1, c0 = np.polyfit(xs, ys, 2)
        if c2 < 1e-12:                       # linear (or concave -> would reward extremes): use a line
            c1, c0 = np.polyfit(xs, ys, 1)
            c2 = 0.0
        err = float(np.max(np.abs(ys - (c2 * xs ** 2 + c1 * xs + c0))))
        if err > 1e-6 * max(1.0, float(np.max(np.abs(ys)))):
            self.cost_model_warnings.append(
                f"pwl_cost of {label} approximated by a quadratic (max abs error {err:.4g} EUR)")
        return (float(c0), float(c1), float(c2))

    def _build_cost_table(self, net):
        self.cost_model_warnings = []
        table = {et: {} for et in self._DISPATCH_TYPES}

        if hasattr(net, 'poly_cost') and not net.poly_cost.empty:
            for _, row in net.poly_cost.iterrows():
                et = row['et']
                if et not in table:
                    continue
                table[et][int(row['element'])] = (
                    self._val(row.get('cp0_eur'), 0.0),
                    self._val(row.get('cp1_eur_per_mw'), 1.0),
                    self._val(row.get('cp2_eur_per_mw2'), 0.0))

        if hasattr(net, 'pwl_cost') and not net.pwl_cost.empty:
            for _, row in net.pwl_cost.iterrows():
                et, el = row['et'], int(row['element'])
                if et not in table or el in table[et]:
                    continue                                   # poly_cost wins over pwl_cost
                if row.get('power_type', 'p') != 'p':
                    continue                                   # reactive-power costs are not part of DC-OPF
                if el not in getattr(net, et).index:
                    continue
                p_lo, p_hi = self._asset_bounds(net, et, el)
                table[et][el] = self._fit_pwl_quadratic(row['points'], p_lo, p_hi, f"{et} {el}")

        self._fallback_c1 = max([c[1] for c in table['gen'].values()] + [100.0])
        self._costs = table

    def _cost_coeffs(self, et, idx):
        coeffs = self._costs[et].get(int(idx))
        if coeffs is not None:
            return coeffs
        return (0.0, self._fallback_c1 * (10.0 if et == 'ext_grid' else 1.0), 0.0)

    # ------------------------------------------------------------------ #
    # Network preprocessing
    # ------------------------------------------------------------------ #
    def _get_line_physics(self, net, line_idx):
        line = net.line.loc[line_idx]
        f_bus = int(line['from_bus'])
        t_bus = int(line['to_bus'])
        vn_kv = net.bus.at[f_bus, 'vn_kv']
        sn_mva = getattr(net, 'sn_mva', 100.0)

        z_base = (vn_kv ** 2) / sn_mva
        x_ohm = line['x_ohm_per_km'] * line['length_km'] / self._val(line.get('parallel'), 1.0)
        x_pu = x_ohm / z_base
        b_mw_rad = sn_mva / x_pu if x_pu != 0 else 0.0

        max_i_ka = self._val(line.get('max_i_ka'), None)
        max_loading = self._val(line.get('max_loading_percent'), 100.0) / 100.0   # NaN-safe

        if max_i_ka is None or max_i_ka > self.UNCONSTRAINED_I_KA:
            p_max_mw = 0.0
        else:
            p_max_mw = math.sqrt(3) * vn_kv * max_i_ka * max_loading
            if p_max_mw > self.UNCONSTRAINED_MW:
                p_max_mw = 0.0

        return f_bus, t_bus, b_mw_rad, p_max_mw

    def _prepare_network(self, net):
        # in-service buses
        self._active_buses = {int(b) for b in net.bus.index[net.bus.in_service.astype(bool)]}

        # in-service loads (with scaling), aggregated per bus
        self._load_mw = {}
        if not net.load.empty:
            for _, ld in net.load[net.load.in_service.astype(bool)].iterrows():
                bus = int(ld['bus'])
                self._load_mw[bus] = self._load_mw.get(bus, 0.0) + ld['p_mw'] * self._val(ld.get('scaling'), 1.0)

        # active lines with physics (cached: used by encoding, balance, limits, decode, printing)
        self._lines = []
        self._max_b = {}
        for l_idx in net.line.index[net.line.in_service.astype(bool)]:
            f, t, b, p_max = self._get_line_physics(net, l_idx)
            if b == 0.0 or f not in self._active_buses or t not in self._active_buses:
                continue
            self._lines.append({'idx': int(l_idx), 'f': f, 't': t, 'b': b, 'p_max': p_max})
            for bus in (f, t):
                self._max_b[bus] = max(self._max_b.get(bus, 0.0), abs(b))

        self._build_cost_table(net)
        self._angle_bounds = self._compute_angle_bounds(net)

    def _compute_angle_bounds(self, net):
        """
        Half-range d_i for bus angle theta_i in [-d_i, d_i] (slack buses fixed at 0).

        |theta_i - theta_slack| <= sum over any path of |flow_l| / B_l, so d_i is the shortest path
        (Dijkstra from the slack buses) with edge weight  flow_bound_l / B_l, where flow_bound_l is the
        line rating if the line is monitored, otherwise the total load (a heuristic bound).
        """
        slack = {int(net.ext_grid.at[i, 'bus']) for i in self._active(net, 'ext_grid')}
        if not slack:
            raise ValueError("DC-OPF needs at least one in-service ext_grid as angle reference.")

        total_load = max(sum(self._load_mw.values()), self.mw_precision)
        adj = {int(b): [] for b in net.bus.index}
        for ln in self._lines:
            cap = ln['p_max'] if (self.enforce_line_limits and ln['p_max'] > 0.0) else total_load
            w = cap / abs(ln['b'])
            adj[ln['f']].append((ln['t'], w))
            adj[ln['t']].append((ln['f'], w))

        dist = {b: 0.0 for b in slack}
        heap = [(0.0, b) for b in slack]
        heapq.heapify(heap)
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, math.inf):
                continue
            for v, w in adj.get(u, []):
                if d + w < dist.get(v, math.inf):
                    dist[v] = d + w
                    heapq.heappush(heap, (d + w, v))

        bounds = {}
        for b, neighbours in adj.items():
            if b in slack or not neighbours:
                bounds[b] = 0.0                                   # reference, or angle irrelevant (no lines)
            elif b in dist:
                bounds[b] = dist[b] * self.ANGLE_HEADROOM
            else:
                bounds[b] = self.fallback_angle_rad               # no path to a slack bus
        return bounds

    # ------------------------------------------------------------------ #
    # Variable encoding
    # ------------------------------------------------------------------ #
    def _encode_variables(self, net):
        self.var_registry = {'gen': {}, 'sgen': {}, 'ext_grid': {}, 'bus': {}, 'slack_lines': {}}
        total_qubits = 0

        for et in self._DISPATCH_TYPES:
            for idx in self._active(net, et):
                p_min, p_max = self._asset_bounds(net, et, idx)
                total_qubits += self._register(et, int(idx), f"{self._BIT_PREFIX[et]}_{idx}",
                                               p_min, p_max, self.mw_precision)

        for bus in net.bus.index:
            bus = int(bus)
            d = self._angle_bounds.get(bus, 0.0)
            max_b = self._max_b.get(bus, 0.0)
            if d > 0.0 and max_b > 0.0:
                prec = self.angle_precision if self.angle_precision is not None else self.mw_precision / max_b
                total_qubits += self._register('bus', bus, f"bus_{bus}", -d, d, prec)
            else:
                total_qubits += self._register('bus', bus, f"bus_{bus}", 0.0, 0.0, 0.0)

        return total_qubits

    def _resolve_penalties(self):
        """
        Violating a constraint by delta MW can save at most (max marginal cost) * delta of objective, while
        it costs lambda * delta^2 in penalty. lambda >= c_max / precision therefore makes every violation of
        at least one MW grid step unprofitable; `penalty_safety` adds headroom.
        """
        marginals = []
        for et in self._DISPATCH_TYPES:
            for idx, reg in self.var_registry[et].items():
                _, c1, c2 = self._cost_coeffs(et, idx)
                marginals.append(abs(c1) + 2.0 * abs(c2) * reg['max'])
        c_max = max(marginals + [1.0])
        auto = self.penalty_safety * c_max / self.mw_precision
        self.lambda_balance = self.penalty_balance if self.penalty_balance is not None else auto
        self.lambda_line = self.penalty_line if self.penalty_line is not None else auto

    # ------------------------------------------------------------------ #
    # QUBO terms
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_polynomial_cost(bqm, reg, c0, c1, c2):
        """Adds c2*P^2 + c1*P + c0 with P = p_min + sum_k w_k b_k."""
        p_min, bits, weights = reg['min'], reg['bits'], reg['weights']
        bqm.offset += (c2 * p_min ** 2) + (c1 * p_min) + c0
        for k, bit in enumerate(bits):
            bqm.add_linear(bit, (2 * c2 * p_min * weights[k]) + (c2 * weights[k] ** 2) + (c1 * weights[k]))
        for i in range(len(bits)):
            for j in range(i + 1, len(bits)):
                bqm.add_quadratic(bits[i], bits[j], 2 * c2 * weights[i] * weights[j])

    def _build_objective(self, bqm, net):
        for et in self._DISPATCH_TYPES:
            for idx, reg in self.var_registry[et].items():       # registry only holds in-service assets
                c0, c1, c2 = self._cost_coeffs(et, idx)
                self._add_polynomial_cost(bqm, reg, c0, c1, c2)
        return bqm

    def _add_squared_penalty(self, bqm, terms_dict, constant, penalty_weight):
        """Adds penalty_weight * (constant + sum_i a_i x_i)^2."""
        terms = [(bit, w) for bit, w in terms_dict.items() if w != 0.0]
        bqm.offset += penalty_weight * (constant ** 2)
        for bit, w in terms:
            bqm.add_linear(bit, penalty_weight * ((2 * constant * w) + (w ** 2)))
        for i in range(len(terms)):
            bi, wi = terms[i]
            for j in range(i + 1, len(terms)):
                bj, wj = terms[j]
                bqm.add_quadratic(bi, bj, penalty_weight * 2 * wi * wj)

    def _build_power_balance(self, bqm, net):
        """KCL at every in-service bus: gen + sgen + ext - load - sum(outgoing line flows) = 0."""
        terms = {b: {} for b in self._active_buses}
        const = {b: -self._load_mw.get(b, 0.0) for b in self._active_buses}

        def add(bus, reg, coeff):
            const[bus] += coeff * reg['min']
            t = terms[bus]
            for bit, w in zip(reg['bits'], reg['weights']):
                t[bit] = t.get(bit, 0.0) + coeff * w

        for et in self._DISPATCH_TYPES:
            tbl = getattr(net, et)
            for idx, reg in self.var_registry[et].items():
                bus = int(tbl.at[idx, 'bus'])
                if bus in terms:
                    add(bus, reg, 1.0)

        for ln in self._lines:                                    # flow f->t = B (theta_f - theta_t)
            f, t, b = ln['f'], ln['t'], ln['b']
            reg_f, reg_t = self.var_registry['bus'][f], self.var_registry['bus'][t]
            add(f, reg_f, -b); add(f, reg_t, +b)                  # leaves f
            add(t, reg_f, +b); add(t, reg_t, -b)                  # enters t

        for bus in sorted(self._active_buses):
            self._add_squared_penalty(bqm, terms[bus], const[bus], self.lambda_balance)
        return bqm

    def _build_line_limits(self, bqm, net):
        """|flow| <= p_max  <=>  p_max + flow - s = 0 with slack s in [0, 2 p_max]."""
        for ln in self._lines:
            p_max = ln['p_max']
            if p_max <= 0.0:
                continue
            f, t, b = ln['f'], ln['t'], ln['b']
            reg_f, reg_t = self.var_registry['bus'][f], self.var_registry['bus'][t]

            terms = {}
            constant = p_max + b * reg_f['min'] - b * reg_t['min']
            for k, bit in enumerate(reg_f['bits']):
                terms[bit] = terms.get(bit, 0.0) + b * reg_f['weights'][k]
            for k, bit in enumerate(reg_t['bits']):
                terms[bit] = terms.get(bit, 0.0) - b * reg_t['weights'][k]

            self._register('slack_lines', ln['idx'], f"slack_line_{ln['idx']}", 0.0, 2.0 * p_max, self.mw_precision)
            slack = self.var_registry['slack_lines'][ln['idx']]
            for bit, w in zip(slack['bits'], slack['weights']):
                terms[bit] = -w

            self._add_squared_penalty(bqm, terms, constant, self.lambda_line)
        return bqm

    # ------------------------------------------------------------------ #
    # Formulation / decoding
    # ------------------------------------------------------------------ #
    def _formulate_qubo(self, net):
        self._prepare_network(net)
        self._encode_variables(net)
        self._resolve_penalties()

        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
        bqm = self._build_objective(bqm, net)
        bqm = self._build_power_balance(bqm, net)
        if self.enforce_line_limits:
            bqm = self._build_line_limits(bqm, net)

        num_slack_qubits = sum(len(r['bits']) for r in self.var_registry['slack_lines'].values())
        num_angle_qubits = sum(len(r['bits']) for r in self.var_registry['bus'].values())
        n_monitored = sum(1 for ln in self._lines if ln['p_max'] > 0.0)

        complexity = {
            "classical_domain_input": {
                "continuous_variables": sum(len(self.var_registry[et]) for et in self._DISPATCH_TYPES)
                                        + sum(1 for r in self.var_registry['bus'].values() if r['bits']),
                "equality_constraints": len(self._active_buses),
                "inequality_constraints": 2 * n_monitored if self.enforce_line_limits else 0,
            },
            "quantum_domain_qubo": {
                "total_logical_qubits": len(bqm.variables),
                "qubits_used_for_variables": len(bqm.variables) - num_slack_qubits,
                "qubits_used_for_angles": num_angle_qubits,
                "qubits_wasted_on_slack": num_slack_qubits,
                "num_interactions": bqm.num_interactions,
                "offset": bqm.offset,
            },
            "penalties": {"balance": self.lambda_balance, "line": self.lambda_line},
        }
        return bqm, complexity

    def _decode_solution(self, sample, net):
        reg_all = self.var_registry
        raw = {et: {idx: self._decode_value(reg, sample) for idx, reg in reg_all[et].items()}
               for et in self._DISPATCH_TYPES}
        angles = {b: self._decode_value(reg, sample) for b, reg in reg_all['bus'].items()}

        # --- nodal balance (local KCL) + bound check ---
        mismatch = {b: 0.0 for b in self._active_buses}
        bounds_ok = True
        for et in self._DISPATCH_TYPES:
            tbl = getattr(net, et)
            for idx, val in raw[et].items():
                reg = reg_all[et][idx]
                if val < reg['min'] - 1e-6 or val > reg['max'] + 1e-6:
                    bounds_ok = False
                bus = int(tbl.at[idx, 'bus'])
                if bus in mismatch:
                    mismatch[bus] += val
        for bus, load in self._load_mw.items():
            if bus in mismatch:
                mismatch[bus] -= load

        # --- line flows and limits ---
        flows, max_line_violation = {}, 0.0
        for ln in self._lines:
            flow = ln['b'] * (angles[ln['f']] - angles[ln['t']])
            flows[ln['idx']] = round(float(flow), 3)
            mismatch[ln['f']] -= flow
            mismatch[ln['t']] += flow
            if ln['p_max'] > 0.0:
                max_line_violation = max(max_line_violation, abs(flow) - ln['p_max'])

        # --- feasibility ---
        tol = self.feasibility_tol_mw if self.feasibility_tol_mw is not None else self.mw_precision
        max_nodal = max((abs(m) for m in mismatch.values()), default=0.0)
        nodal_ok = max_nodal <= tol
        lines_ok = (not self.enforce_line_limits) or (max_line_violation <= tol)

        total_gen = sum(sum(v.values()) for v in raw.values())
        feasibility = {
            "total_generation_mw": round(total_gen, 3),
            "total_load_mw": round(sum(self._load_mw.values()), 3),
            "max_nodal_mismatch_mw": round(float(max_nodal), 3),
            "max_line_violation_mw": round(float(max_line_violation), 3),
            "tolerance_mw": tol,
            "nodal_ok": bool(nodal_ok),
            "lines_ok": bool(lines_ok),
            "bounds_ok": bool(bounds_ok),
            "is_feasible": bool(nodal_ok and lines_ok and bounds_ok),
            "line_flows_mw": flows,
            "bus_angles_rad": {b: round(float(a), 6) for b, a in angles.items()},
        }

        # --- cost, from the same coefficients the QUBO objective used ---
        cost = 0.0
        for et in self._DISPATCH_TYPES:
            for idx, val in raw[et].items():
                c0, c1, c2 = self._cost_coeffs(et, idx)
                cost += (c2 * val ** 2) + (c1 * val) + c0

        dispatch = {i: {'p_mw': round(v, 3)} for i, v in raw['gen'].items()}
        sgen_dispatch = {i: {'p_mw': round(v, 3)} for i, v in raw['sgen'].items()}
        slack_dispatch = {i: {'p_mw': round(v, 3)} for i, v in raw['ext_grid'].items()}
        return dispatch, sgen_dispatch, slack_dispatch, round(cost, 3), feasibility

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def view_problem_definition(self, net):
        bqm, complexity = self._formulate_qubo(net)
        self._print_problem_parameters(net, complexity)
        return bqm, complexity

    @staticmethod
    def _fmt_weights(weights, max_show=10):
        w = [f"{x:.4g}" for x in weights]
        if len(w) > max_show:
            w = w[:4] + ["..."] + w[-2:]
        return ", ".join(w)

    def _variable_symbol(self, group, idx):
        return {"gen": f"P_gen{idx}", "sgen": f"P_sgen{idx}", "ext_grid": f"P_ext{idx}",
                "bus": f"theta_{idx}"}[group]

    def _print_problem_parameters(self, net, complexity):
        line = "=" * 80
        print("\n" + line)
        print(f"   QUBO {self.formulation.upper()}-OPF: COMPLETE PROBLEM DEFINITION")
        print(line)
        print(f"Total Grid Demand (Load): {round(sum(self._load_mw.values()), 3)} MW")
        print(f"MW precision: {self.mw_precision} MW | Line limits enforced: {self.enforce_line_limits}\n")

        labels = {'gen': 'Gen', 'sgen': 'SGen', 'ext_grid': 'Ext_Grid (Slack)'}

        # 1) objective ------------------------------------------------------
        print("--- OBJECTIVE: minimise total generation cost ---")
        for et in self._DISPATCH_TYPES:
            tbl = getattr(net, et)
            for idx in self.var_registry[et]:
                c0, c1, c2 = self._cost_coeffs(et, idx)
                sym = self._variable_symbol(et, idx)
                note = "" if idx in self._costs[et] else "   [fallback cost: none given in net]"
                print(f" {labels[et]} {idx} (Bus {tbl.at[idx, 'bus']}): "
                      f"{c2:.6g}*{sym}^2 + {c1:.6g}*{sym} + {c0:.6g}{note}")
        for w in self.cost_model_warnings:
            print(f" [cost warning] {w}")
        print(" QUBO energy = sum(costs) + lambda_balance * sum_buses(KCL residual)^2"
              + (" + lambda_line * sum_lines(line residual)^2" if self.enforce_line_limits else ""))

        # 2) variables ------------------------------------------------------
        print("\n--- DECISION VARIABLES (binary encoding: value = min + sum_k w_k * bit_k) ---")
        for et in self._DISPATCH_TYPES:
            for idx, reg in self.var_registry[et].items():
                sym = self._variable_symbol(et, idx)
                print(f" {sym:<10} in [{reg['min']}, {reg['max']}] MW | {len(reg['bits']):>2} bits | "
                      f"weights: {self._fmt_weights(reg['weights'])}")
        for bus, reg in self.var_registry['bus'].items():
            sym = self._variable_symbol('bus', bus)
            if reg['bits']:
                print(f" {sym:<10} in [{reg['min']:.4f}, {reg['max']:.4f}] rad | {len(reg['bits']):>2} bits | "
                      f"step {reg['precision']:.3g} rad")
            else:
                print(f" {sym:<10} fixed at 0 (slack reference or no lines)")
        for l_idx, reg in self.var_registry['slack_lines'].items():
            print(f" s_line{l_idx:<4} in [0, {reg['max']:.4g}] MW | {len(reg['bits']):>2} bits | "
                  f"(inequality slack, not a physical variable)")

        # 3) network and constraints ---------------------------------------
        print("\n--- TRANSMISSION LINES (DC-OPF: flow_ij = B_ij * (theta_i - theta_j)) ---")
        for ln in self._lines:
            limit_str = f"|flow| <= {round(ln['p_max'], 2)} MW" if ln['p_max'] > 0 else "unconstrained"
            print(f" Line {ln['idx']} ({ln['f']} -> {ln['t']}): B = {round(ln['b'], 2)} MW/rad, {limit_str}")

        print("\n--- CONSTRAINTS (soft, as quadratic penalties) ---")
        for bus in sorted(self._active_buses):
            inj = [self._variable_symbol(et, idx) for et in self._DISPATCH_TYPES
                   for idx in self.var_registry[et] if int(getattr(net, et).at[idx, 'bus']) == bus]
            flows = []
            for ln in self._lines:
                if ln['f'] == bus:
                    flows.append(f"flow({ln['f']}->{ln['t']})")
                elif ln['t'] == bus:
                    flows.append(f"flow({ln['t']}->{ln['f']})")
            load = round(self._load_mw.get(bus, 0.0), 3)
            print(f" KCL bus {bus}: {' + '.join(inj) if inj else '0'} - {load} MW load "
                  f"- outgoing[{', '.join(flows)}] = 0")
        if self.enforce_line_limits:
            for l_idx in self.var_registry['slack_lines']:
                ln = next(x for x in self._lines if x['idx'] == l_idx)
                print(f" Line {l_idx}: {round(ln['p_max'], 2)} + flow({ln['f']}->{ln['t']}) - s_line{l_idx} = 0")
        auto_b = "auto" if self.penalty_balance is None else "manual"
        auto_l = "auto" if self.penalty_line is None else "manual"
        print(f" lambda_balance = {self.lambda_balance:.6g} ({auto_b}) | lambda_line = {self.lambda_line:.6g} ({auto_l})")

        # 4) size summary ---------------------------------------------------
        c, q = complexity['classical_domain_input'], complexity['quantum_domain_qubo']
        print("\n--- PROBLEM SIZE ---")
        print(f" Classical: {c['continuous_variables']} continuous variables, "
              f"{c['equality_constraints']} equality and {c['inequality_constraints']} inequality constraints")
        print(f" QUBO: {q['total_logical_qubits']} logical qubits "
              f"(dispatch {q['qubits_used_for_variables'] - q['qubits_used_for_angles']}, "
              f"angles {q['qubits_used_for_angles']}, line-limit slack {q['qubits_wasted_on_slack']}), "
              f"{q['num_interactions']} quadratic terms, constant offset {q['offset']:.6g}")
        print(line + "\n")


if __name__ == "__main__":
    # Demo: 3-bus, 110 kV meshed grid. Cheap local gen (quadratic cost) at bus 1,
    # expensive ext_grid (slack) at bus 0, 50.5 MW load at bus 2.
    import pandapower as pp

    net = pp.create_empty_network(sn_mva=100.0)
    buses = [pp.create_bus(net, vn_kv=110.0) for _ in range(3)]
    pp.create_ext_grid(net, buses[0], min_p_mw=0.0, max_p_mw=100.0)
    pp.create_gen(net, buses[1], p_mw=0.0, min_p_mw=0.0, max_p_mw=60.0, controllable=True)
    pp.create_load(net, buses[2], p_mw=50.5)
    for f_bus, t_bus in [(0, 1), (1, 2), (0, 2)]:
        pp.create_line_from_parameters(net, buses[f_bus], buses[t_bus], length_km=1.0, r_ohm_per_km=0.1,
                                       x_ohm_per_km=30.0, c_nf_per_km=0.0, max_i_ka=0.4)
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=50.0)
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=10.0, cp2_eur_per_mw2=0.05)

    formulator = BaseQuboFormulator(mw_precision=1.0)
    bqm, complexity = formulator.view_problem_definition(net)

    # Optional: solve with simulated annealing (pip install dwave-neal) and decode the best sample.
    try:
        import neal
    except ImportError:
        print("Tip: `pip install dwave-neal` to also solve this QUBO with SA in the demo.")
    else:
        best = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=200, num_sweeps=3000, seed=1).first
        gen, sgen, ext, cost, feas = formulator._decode_solution(best.sample, net)
        print("--- SA RESULT (best of 200 reads) ---")
        print(f" QUBO energy : {best.energy:.4f}")
        print(f" Gen dispatch: {gen}")
        print(f" Ext dispatch: {ext}")
        print(f" Cost        : {cost} EUR")
        print(f" Feasible    : {feas['is_feasible']} (max nodal mismatch {feas['max_nodal_mismatch_mw']} MW, "
              f"max line violation {feas['max_line_violation_mw']} MW)")
        print(f" Line flows  : {feas['line_flows_mw']}")