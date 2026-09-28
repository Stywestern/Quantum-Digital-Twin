import heapq
import math

import dimod
import numpy as np

class QuboFormulator():
    # Line ratings that are missing / absurdly large are treated as "unconstrained".
    UNCONSTRAINED_I_KA = 100.0
    UNCONSTRAINED_MW = 1000.0
    # Dispatch upper bound used when an asset has no max_p_mw.
    DEFAULT_MAX_MW = 500.0
    # dc_theta: angle ranges are widened by this factor so penalised (slightly violating) states stay representable.
    ANGLE_HEADROOM = 1.1

    FORMULATIONS = ("dc_ptdf", "dc_theta")
    ENCODINGS = ("radix", "unary", "hybrid")
    _DISPATCH_TYPES = ("gen", "sgen", "ext_grid")
    _BIT_PREFIX = {"gen": "gen", "sgen": "sgen", "ext_grid": "ext"}

    def __init__(self, formulation="dc", mw_precision=1.0, 
                 angle_precision=0.01, penalty_balance=None, penalty_line=None,
                 encoding="radix", penalty_safety=1.1, ptdf_threshold=1e-4,
                 hybrid_chunk_size=50.0, feasibility_tol_mw=None, rebalance_slack=True,
                 scale_line_constraints=True, smart_slack_side=True,
                 slack_precision_factor=1.0, decode_with_full_ptdf=True,
                 ptdf_rel_threshold=None, ptdf_round_to=None, snap_weights=True,
                 ext_fallback_mult=10.0, risk_noise_frac=0.02, **kwargs):
        """
        formulation         : "dc_ptdf" (default; "dc" is an alias) or "dc_theta".
        encoding             : "radix", "unary", "hybrid"
        angle_precision     : dc_theta only.
        penalty_balance/line: None -> auto derived.
        hybrid_chunk_size    : Size of unary chunks in hybrid encoding (MW).
        scale_line_constraints: Divides line constraints by max PTDF to shrink dynamic range.
        smart_slack_side     : Halves slack bit requirements by bounding strictly.
        ptdf_rel_threshold   : Zero PTDF entries below rel * max|H_l| to sparsify the graph.
        """

        if formulation == "dc":
            formulation = "dc_ptdf"
        if formulation == "ac":
            raise NotImplementedError("The AC formulation is not implemented.")
        if formulation not in self.FORMULATIONS:
            raise ValueError(f"formulation must be one of {self.FORMULATIONS} (or 'dc'), got {formulation!r}")
        if encoding not in self.ENCODINGS:
            raise ValueError(f"encoding must be one of {self.ENCODINGS}, got {encoding!r}")

        self.formulation = formulation
        self.mw_precision = mw_precision
        self.angle_precision = angle_precision
        self.penalty_balance = penalty_balance
        self.penalty_line = penalty_line
        self.encoding = encoding
        self.penalty_safety = penalty_safety
        self.ptdf_threshold = ptdf_threshold
        self.hybrid_chunk_size = hybrid_chunk_size
        self.feasibility_tol_mw = feasibility_tol_mw
        self.rebalance_slack = rebalance_slack
        
        # New Scaled Parameters
        self.scale_line_constraints = scale_line_constraints
        self.smart_slack_side = smart_slack_side
        self.slack_precision_factor = slack_precision_factor
        self.decode_with_full_ptdf = decode_with_full_ptdf
        self.ptdf_rel_threshold = ptdf_rel_threshold
        self.ptdf_round_to = ptdf_round_to
        self.snap_weights = snap_weights
        self.ext_fallback_mult = ext_fallback_mult
        self.risk_noise_frac = risk_noise_frac

        self.var_registry = {}
        self.cost_model_warnings = []
        self.lambda_balance = penalty_balance
        self.lambda_line = penalty_line

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _val(x, default):
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
            raise ValueError(f"{et} {idx}: max_p_mw ({p_max}) < min_p_mw ({p_min}); check the limits.")
        return p_min, p_max

    def _get_radix_weights(self, total_range, precision):
        if total_range <= 1e-12:
            return []
        n = max(1, int(math.floor(total_range / precision + 1e-9)))
        K = int(math.floor(math.log2(n)))
        weights = [precision * (2 ** k) for k in range(K)]
        weights.append(total_range - (2 ** K - 1) * precision)
        return weights

    def _get_unary_weights(self, total_range, precision):
        if total_range <= 1e-12:
            return []
        n = max(1, int(math.floor(total_range / precision + 1e-9)))
        weights = [precision] * n
        remainder = total_range - n * precision
        if remainder > 1e-12:
            weights.append(remainder)
        return weights

    def _get_hybrid_weights(self, total_range, precision):
        if total_range <= 1e-12:
            return []
        if total_range <= self.hybrid_chunk_size:
            return self._get_radix_weights(total_range, precision)
            
        radix_coverage = self.hybrid_chunk_size - precision
        radix_weights = self._get_radix_weights(radix_coverage, precision)
        
        remainder = total_range - radix_coverage
        n_full_chunks = int(remainder // self.hybrid_chunk_size)
        leftover_chunk = remainder % self.hybrid_chunk_size
        
        unary_weights = [self.hybrid_chunk_size] * n_full_chunks
        if leftover_chunk > 1e-6:
            unary_weights.append(leftover_chunk)
            
        return unary_weights + radix_weights

    def _get_weights(self, total_range, precision):
        if total_range <= 0:
            return []
            
        weights = []
        if self.encoding == "radix":
            weights = self._get_radix_weights(total_range, precision)
        elif self.encoding == "unary":
            weights = self._get_unary_weights(total_range, precision)
        elif self.encoding == "hybrid":
            weights = self._get_hybrid_weights(total_range, precision)
                
        # Apply weight snapping
        if not self.snap_weights or not precision or precision <= 0:
            return weights
            
        snapped = []
        for x in weights:
            k = math.floor(x / precision + 1e-9)
            if k > 0:
                snapped.append(k * precision)
        return snapped if snapped else weights

    def _register(self, group, key, prefix, abs_lo, abs_hi, precision):
        lo, hi = abs_lo, abs_hi
        weights = self._get_weights(hi - lo, precision) if hi > lo else []
            
        bits = [f"{prefix}_bit_{k}" for k in range(len(weights))]
        self.var_registry[group][key] = {
            'min': lo, 'max': hi, 'precision': precision, 'bits': bits, 'weights': weights
        }
        return len(bits)

    @staticmethod
    def _decode_value(reg, sample):
        return reg['min'] + sum(w for bit, w in zip(reg['bits'], reg['weights']) if sample.get(bit, 0) == 1)

    def _dispatch_assets(self, net):
        for et in self._DISPATCH_TYPES:
            tbl = getattr(net, et)
            for idx, reg in self.var_registry[et].items():
                yield et, idx, reg, int(tbl.at[idx, 'bus'])

    # ------------------------------------------------------------------ #
    # Cost model
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_pwl(segments, label):
        segs = []
        for seg in segments:
            if len(seg) != 3:
                raise ValueError(f"pwl_cost of {label}: expected segments (p_from, p_to, slope), got {seg!r}")
            segs.append((float(seg[0]), float(seg[1]), float(seg[2])))
        segs.sort()
        return segs

    @staticmethod
    def _pwl_eval(segs, p):
        p = np.asarray(p, dtype=float)
        total = np.zeros_like(p)
        for k, (a, b, slope) in enumerate(segs):
            if k == len(segs) - 1:
                total += slope * np.maximum(p - a, 0.0)
            else:
                total += slope * np.clip(p - a, 0.0, b - a)
        return total

    def _fit_quadratic(self, segs, p_lo, p_hi, label):
        if p_hi <= p_lo:
            return (float(self._pwl_eval(segs, [p_lo])[0]), 0.0, 0.0)
        xs = np.linspace(p_lo, p_hi, 201)
        ys = self._pwl_eval(segs, xs)
        c2, c1, c0 = np.polyfit(xs, ys, 2)
        if c2 < 1e-12:
            c1, c0 = np.polyfit(xs, ys, 1)
            c2 = 0.0
        err = float(np.max(np.abs(ys - (c2 * xs ** 2 + c1 * xs + c0))))
        if err > 1e-6 * max(1.0, float(np.max(np.abs(ys)))):
            self.cost_model_warnings.append(
                f"pwl_cost of {label} approximated by a quadratic inside the QUBO; reported costs use exact PWL")
        return (float(c0), float(c1), float(c2))

    def _build_cost_table(self, net):
        self.cost_model_warnings = []
        table = {et: {} for et in self._DISPATCH_TYPES}

        if hasattr(net, 'poly_cost') and not net.poly_cost.empty:
            for _, row in net.poly_cost.iterrows():
                et = row['et']
                if et not in table:
                    continue
                table[et][int(row['element'])] = {'pwl': None, 'coeffs': (
                    self._val(row.get('cp0_eur'), 0.0),
                    self._val(row.get('cp1_eur_per_mw'), 1.0),
                    self._val(row.get('cp2_eur_per_mw2'), 0.0))}

        if hasattr(net, 'pwl_cost') and not net.pwl_cost.empty:
            for _, row in net.pwl_cost.iterrows():
                et, el = row['et'], int(row['element'])
                if et not in table or el in table[et]:
                    continue
                if row.get('power_type', 'p') != 'p':
                    continue
                if el not in getattr(net, et).index:
                    continue
                p_lo, p_hi = self._asset_bounds(net, et, el)
                segs = self._parse_pwl(row['points'], f"{et} {el}")
                table[et][el] = {'pwl': segs, 'coeffs': self._fit_quadratic(segs, p_lo, p_hi, f"{et} {el}")}

        self._fallback_c1 = max([e['coeffs'][1] for e in table['gen'].values()] + [100.0])
        self._costs = table

    def _cost_coeffs(self, et, idx):
        entry = self._costs[et].get(int(idx))
        if entry is not None:
            return entry['coeffs']
        return (0.0, self._fallback_c1 * (getattr(self, 'ext_fallback_mult', 10.0) if et == 'ext_grid' else 1.0), 0.0)

    def _true_cost(self, et, idx, p):
        entry = self._costs[et].get(int(idx))
        if entry is not None and entry['pwl'] is not None:
            return float(self._pwl_eval(entry['pwl'], [p])[0])
        c0, c1, c2 = self._cost_coeffs(et, idx)
        return c2 * p * p + c1 * p + c0

    def _total_true_cost(self, values):
        return sum(self._true_cost(et, idx, v) for et in self._DISPATCH_TYPES for idx, v in values[et].items())

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
        max_loading = self._val(line.get('max_loading_percent'), 100.0) / 100.0

        if max_i_ka is None or max_i_ka > self.UNCONSTRAINED_I_KA:
            p_max_mw = 0.0
        else:
            p_max_mw = math.sqrt(3) * vn_kv * max_i_ka * max_loading
            if p_max_mw > self.UNCONSTRAINED_MW:
                p_max_mw = 0.0

        return f_bus, t_bus, b_mw_rad, p_max_mw

    def _get_trafo_physics(self, net, trafo_idx):
        trafo = net.trafo.loc[trafo_idx]
        f_bus = int(trafo['hv_bus'])
        t_bus = int(trafo['lv_bus'])
        
        sn_mva_trafo = trafo['sn_mva']
        vk_percent = trafo['vk_percent']
        parallel = self._val(trafo.get('parallel'), 1.0)
        
        if vk_percent == 0:
            b_mw_rad = 0.0
        else:
            b_mw_rad = (sn_mva_trafo * parallel * 100.0) / vk_percent

        max_loading = self._val(trafo.get('max_loading_percent'), 100.0) / 100.0
        p_max_mw = sn_mva_trafo * parallel * max_loading
        
        return f_bus, t_bus, b_mw_rad, p_max_mw

    def _prepare_network(self, net):
        self._active_buses = {int(b) for b in net.bus.index[net.bus.in_service.astype(bool)]}

        self._load_mw = {}
        if not net.load.empty:
            for _, ld in net.load[net.load.in_service.astype(bool)].iterrows():
                bus = int(ld['bus'])
                if bus in self._active_buses:
                    self._load_mw[bus] = self._load_mw.get(bus, 0.0) + ld['p_mw'] * self._val(ld.get('scaling'), 1.0)

        self._lines = []
        self._max_b = {}
        
        for l_idx in net.line.index[net.line.in_service.astype(bool)]:
            f, t, b, p_max = self._get_line_physics(net, l_idx)
            if b == 0.0 or f not in self._active_buses or t not in self._active_buses:
                continue
            self._lines.append({'idx': f"line_{l_idx}", 'f': f, 't': t, 'b': b, 'p_max': p_max})
            for bus in (f, t):
                self._max_b[bus] = max(self._max_b.get(bus, 0.0), abs(b))
                
        if hasattr(net, 'trafo') and not net.trafo.empty:
            for t_idx in net.trafo.index[net.trafo.in_service.astype(bool)]:
                f, t, b, p_max = self._get_trafo_physics(net, t_idx)
                if b == 0.0 or f not in self._active_buses or t not in self._active_buses:
                    continue
                self._lines.append({'idx': f"trafo_{t_idx}", 'f': f, 't': t, 'b': b, 'p_max': p_max})
                for bus in (f, t):
                    self._max_b[bus] = max(self._max_b.get(bus, 0.0), abs(b))

        self._build_cost_table(net)
        self._build_ptdf(net)
        if self.formulation == "dc_theta":
            self._angle_bounds = self._compute_angle_bounds(net)

    def _build_ptdf(self, net):
        ext = self._active(net, 'ext_grid')
        if not ext:
            raise ValueError("DC-OPF needs at least one in-service ext_grid as angle reference.")
        self._ref_ext = int(ext[0])
        self._ref_bus = int(net.ext_grid.at[self._ref_ext, 'bus'])
        if self._ref_bus not in self._active_buses:
            raise ValueError(f"ext_grid {self._ref_ext} sits on an out-of-service bus.")

        buses = sorted(self._active_buses)
        self._bus_pos = pos = {b: i for i, b in enumerate(buses)}
        n = len(buses)

        adj = {b: [] for b in buses}
        for ln in self._lines:
            adj[ln['f']].append(ln['t'])
            adj[ln['t']].append(ln['f'])
        seen, stack = {self._ref_bus}, [self._ref_bus]
        while stack:
            for v in adj[stack.pop()]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        missing = sorted(set(buses) - seen)
        if missing:
            raise ValueError(
                f"Buses {missing} are not connected to the reference bus {self._ref_bus}. Only lines/trafos modelled.")

        Bm = np.zeros((n, n))
        for ln in self._lines:
            i, j, b = pos[ln['f']], pos[ln['t']], ln['b']
            Bm[i, i] += b
            Bm[j, j] += b
            Bm[i, j] -= b
            Bm[j, i] -= b
        keep = [i for i in range(n) if i != pos[self._ref_bus]]
        X = np.zeros((n, n))
        if keep:
            try:
                X[np.ix_(keep, keep)] = np.linalg.inv(Bm[np.ix_(keep, keep)])
            except np.linalg.LinAlgError as exc:
                try:
                    X[np.ix_(keep, keep)] = np.linalg.pinv(Bm[np.ix_(keep, keep)])
                except Exception:
                    raise ValueError("Reduced susceptance matrix is singular.") from exc
        self._X = X

        # Store full PTDF
        self._H_full = {ln['idx']: ln['b'] * (X[pos[ln['f']]] - X[pos[ln['t']]]) for ln in self._lines}
        self._H = {}
        
        # Apply Sparsity Thresholding
        for ln in self._lines:
            raw = self._H_full[ln['idx']]
            thr = max(self.ptdf_threshold, (self.ptdf_rel_threshold or 0.0) * float(np.max(np.abs(raw))))
            h = np.where(np.abs(raw) < thr, 0.0, raw)
            if self.ptdf_round_to:
                h = np.round(h / self.ptdf_round_to) * self.ptdf_round_to
            self._H[ln['idx']] = h

    def _compute_angle_bounds(self, net):
        slack = {int(net.ext_grid.at[i, 'bus']) for i in self._active(net, 'ext_grid')}
        total_load = max(sum(self._load_mw.values()), self.mw_precision)
        adj = {int(b): [] for b in net.bus.index}
        for ln in self._lines:
            cap = ln['p_max'] if (ln['p_max'] > 0.0) else total_load
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
                bounds[b] = 0.0
            else:
                bounds[b] = dist[b] * self.ANGLE_HEADROOM
        return bounds

    # ------------------------------------------------------------------ #
    # Variable encoding
    # ------------------------------------------------------------------ #
    def _encode_variables(self, net):
        self.var_registry = {'gen': {}, 'sgen': {}, 'ext_grid': {}, 'bus': {}, 'slack_lines': {}}
        total_qubits = 0

        for et in self._DISPATCH_TYPES:
            for idx in self._active(net, et):
                if int(getattr(net, et).at[idx, 'bus']) not in self._active_buses:
                    continue
                p_min, p_max = self._asset_bounds(net, et, idx)
                total_qubits += self._register(et, int(idx), f"{self._BIT_PREFIX[et]}_{idx}",
                                               p_min, p_max, self.mw_precision)

        if self.formulation == "dc_theta":
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

    def _resolve_penalties(self, net):
        bus_costs = {}
        marginals = []
        
        for et, idx, reg, bus in self._dispatch_assets(net):
            _, c1, c2 = self._cost_coeffs(et, idx)
            c_val = abs(c1) + 2.0 * abs(c2) * reg['max']
            marginals.append(c_val)
            bus_costs[bus] = max(bus_costs.get(bus, 0.0), c_val)
            
        c_max_global = max(marginals + [1.0])
        auto_bal = self.penalty_safety * c_max_global / self.mw_precision
        self.lambda_balance = self.penalty_balance if self.penalty_balance is not None else auto_bal
        
        self.lambda_line = {}
        for ln in self._lines:
            if self.penalty_line is not None:
                self.lambda_line[ln['idx']] = self.penalty_line
            else:
                local_c = max(bus_costs.get(ln['f'], 0.0), bus_costs.get(ln['t'], 0.0))
                if local_c < 1.0: 
                    local_c = c_max_global * 0.5 
                self.lambda_line[ln['idx']] = self.penalty_safety * local_c / self.mw_precision

    # ------------------------------------------------------------------ #
    # QUBO terms
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_polynomial_cost(bqm, reg, c0, c1, c2):
        p_min, bits, weights = reg['min'], reg['bits'], reg['weights']
        bqm.offset += (c2 * p_min ** 2) + (c1 * p_min) + c0
        for k, bit in enumerate(bits):
            bqm.add_linear(bit, (2 * c2 * p_min * weights[k]) + (c2 * weights[k] ** 2) + (c1 * weights[k]))
        for i in range(len(bits)):
            for j in range(i + 1, len(bits)):
                bqm.add_quadratic(bits[i], bits[j], 2 * c2 * weights[i] * weights[j])

    def _build_objective(self, bqm, net):
        for et in self._DISPATCH_TYPES:
            for idx, reg in self.var_registry[et].items(): 
                c0, c1, c2 = self._cost_coeffs(et, idx)
                self._add_polynomial_cost(bqm, reg, c0, c1, c2)
        return bqm

    def _add_squared_penalty(self, bqm, terms_dict, constant, penalty_weight):
        terms = [(bit, w) for bit, w in terms_dict.items() if w != 0.0]
        bqm.offset += penalty_weight * (constant ** 2)
        for bit, w in terms:
            bqm.add_linear(bit, penalty_weight * ((2 * constant * w) + (w ** 2)))
        for i in range(len(terms)):
            bi, wi = terms[i]
            for j in range(i + 1, len(terms)):
                bj, wj = terms[j]
                bqm.add_quadratic(bi, bj, penalty_weight * 2 * wi * wj)

    def _add_line_limit_penalty(self, bqm, ln, terms, constant, slack_upper_bound=None):
        if slack_upper_bound is None:
            slack_upper_bound = 2.0 * ln['p_max']
            
        self._register('slack_lines', ln['idx'], f"slack_line_{ln['idx']}", 0.0, slack_upper_bound, self.mw_precision)
        slack = self.var_registry['slack_lines'][ln['idx']]
        for bit, w in zip(slack['bits'], slack['weights']):
            terms[bit] = terms.get(bit, 0.0) - w
        self._add_squared_penalty(bqm, terms, constant, self.lambda_line[ln['idx']])

    # --- dc_ptdf constraints ---
    def _build_power_balance_ptdf(self, bqm, net):
        terms, const = {}, -sum(self._load_mw.values())
        for et, idx, reg, bus in self._dispatch_assets(net):
            const += reg['min']
            for bit, w in zip(reg['bits'], reg['weights']):
                terms[bit] = terms.get(bit, 0.0) + w
        self._add_squared_penalty(bqm, terms, const, self.lambda_balance)
        return bqm

    def _build_line_limits_ptdf(self, bqm, net):
        if not getattr(self, 'scale_line_constraints', True):
            # Fallback to unscaled logic if flag is False
            for ln in self._lines:
                if ln['p_max'] <= 0.0: continue
                h = self._H[ln['idx']]
                base_flow = -sum(h[self._bus_pos[b]] * ld for b, ld in self._load_mw.items())
                max_possible_flow = base_flow
                min_possible_flow = base_flow
                for et, idx, reg, bus in self._dispatch_assets(net):
                    c = h[self._bus_pos[bus]]
                    val1, val2 = c * reg['min'], c * reg['max']
                    max_possible_flow += max(val1, val2)
                    min_possible_flow += min(val1, val2)
                if max_possible_flow <= ln['p_max'] and min_possible_flow >= -ln['p_max']: continue
                slack_upper_bound = ln['p_max'] + max_possible_flow
                if slack_upper_bound > 2.0 * ln['p_max']: slack_upper_bound = 2.0 * ln['p_max']
                terms, const = {}, ln['p_max'] + base_flow
                for et, idx, reg, bus in self._dispatch_assets(net):
                    c = h[self._bus_pos[bus]]
                    const += c * reg['min']
                    for bit, w in zip(reg['bits'], reg['weights']):
                        terms[bit] = terms.get(bit, 0.0) + c * w
                self._add_line_limit_penalty(bqm, ln, terms, const, slack_upper_bound)
            return bqm

        pos = self._bus_pos
        assets = list(self._dispatch_assets(net))
        slack_prec = self.mw_precision * getattr(self, 'slack_precision_factor', 1.0)

        for ln in self._lines:
            p_max = ln['p_max']
            if p_max <= 0.0:
                continue
            h = self._H[ln['idx']]
            base_flow = -sum(h[pos[b]] * ld for b, ld in self._load_mw.items())
            max_f = min_f = base_flow
            coefs, marg = [], []
            
            for et, idx, reg, bus in assets:
                c = h[pos[bus]]
                v1, v2 = c * reg['min'], c * reg['max']
                max_f += max(v1, v2)
                min_f += min(v1, v2)
                if c != 0.0 and reg['bits']:
                    coefs.append(abs(c))
                    entry = self._costs[et].get(int(idx))
                    if entry is not None:
                        _, c1, c2 = entry['coeffs']
                    else:
                        c1 = self._fallback_c1 * (getattr(self, 'ext_fallback_mult', 10.0) if et == 'ext_grid' else 1.0)
                        c2 = 0.0
                    marg.append(abs(c1) + 2.0 * abs(c2) * reg['max'])

            upper = max_f > p_max + 1e-9            
            lower = min_f < -p_max - 1e-9           
            if not (upper or lower):
                continue                            
            if not coefs:
                self.cost_model_warnings.append(f"{ln['idx']}: limit can be violated but no variable moves it; skipped")
                continue

            if upper and lower or (lower and not upper):
                sgn = 1.0
                ub = 2.0 * p_max if (upper and lower) else p_max + max_f
            else:
                sgn = -1.0
                ub = p_max - min_f
                
            if not getattr(self, 'smart_slack_side', True):
                sgn, ub = 1.0, min(p_max + max_f, 2.0 * p_max)
                
            if ub <= 1e-9:
                self.cost_model_warnings.append(f"{ln['idx']}: limit violated in every state; skipped")
                continue

            sigma = max(coefs)                       
            lam = self.penalty_line if self.penalty_line is not None \
                else self.penalty_safety * max(marg) / self.mw_precision
            self.lambda_line[ln['idx']] = lam

            terms, const = {}, p_max / sigma + sgn * base_flow / sigma
            for et, idx, reg, bus in assets:
                c = sgn * h[pos[bus]] / sigma
                if c == 0.0:
                    continue
                const += c * reg['min']
                for bit, w in zip(reg['bits'], reg['weights']):
                    terms[bit] = terms.get(bit, 0.0) + c * w

            self._register('slack_lines', ln['idx'], f"slack_line_{ln['idx']}", 0.0, ub / sigma, slack_prec)
            slack = self.var_registry['slack_lines'][ln['idx']]
            for bit, w in zip(slack['bits'], slack['weights']):
                terms[bit] = terms.get(bit, 0.0) - w
            self._add_squared_penalty(bqm, terms, const, lam)
            
        return bqm

    # --- dc_theta constraints ---
    def _build_power_balance_theta(self, bqm, net):
        terms = {b: {} for b in self._active_buses}
        const = {b: -self._load_mw.get(b, 0.0) for b in self._active_buses}

        def add(bus, reg, coeff):
            const[bus] += coeff * reg['min']
            t = terms[bus]
            for bit, w in zip(reg['bits'], reg['weights']):
                t[bit] = t.get(bit, 0.0) + coeff * w

        for et, idx, reg, bus in self._dispatch_assets(net):
            add(bus, reg, 1.0)

        for ln in self._lines:                                  
            f, t, b = ln['f'], ln['t'], ln['b']
            reg_f, reg_t = self.var_registry['bus'][f], self.var_registry['bus'][t]
            add(f, reg_f, -b); add(f, reg_t, +b)                  
            add(t, reg_f, +b); add(t, reg_t, -b)                  

        for bus in sorted(self._active_buses):
            self._add_squared_penalty(bqm, terms[bus], const[bus], self.lambda_balance)
        return bqm

    def _build_line_limits_theta(self, bqm, net):
        for ln in self._lines:
            if ln['p_max'] <= 0.0:
                continue
            f, t, b = ln['f'], ln['t'], ln['b']
            reg_f, reg_t = self.var_registry['bus'][f], self.var_registry['bus'][t]
            
            max_flow = abs(b) * (reg_f['max'] - reg_t['min'])
            min_flow = -abs(b) * (reg_f['max'] - reg_t['min'])
            
            if max_flow <= ln['p_max'] and min_flow >= -ln['p_max']:
                continue
                
            slack_upper_bound = ln['p_max'] + max_flow
            if slack_upper_bound > 2.0 * ln['p_max']:
                slack_upper_bound = 2.0 * ln['p_max']

            terms = {}
            constant = ln['p_max'] + b * reg_f['min'] - b * reg_t['min']
            for k, bit in enumerate(reg_f['bits']):
                terms[bit] = terms.get(bit, 0.0) + b * reg_f['weights'][k]
            for k, bit in enumerate(reg_t['bits']):
                terms[bit] = terms.get(bit, 0.0) - b * reg_t['weights'][k]
            self._add_line_limit_penalty(bqm, ln, terms, constant, slack_upper_bound)
        return bqm

    # ------------------------------------------------------------------ #
    # Formulation / decoding
    # ------------------------------------------------------------------ #
    def _range_metrics(self, bqm):
        import dimod
        ising = bqm.change_vartype(dimod.SPIN, inplace=False)
        mags = [abs(b) for b in ising.linear.values() if abs(b) > 1e-12]
        mags += [abs(b) for _, _, b in ising.iter_quadratic() if abs(b) > 1e-12]
        if not mags:
            return {}
        mx, mn = max(mags), min(mags)
        frac = getattr(self, 'risk_noise_frac', 0.02)
        risk = sum(1 for m in mags if m < frac * mx) / len(mags)
        return {"ising_max": round(mx, 4), "ising_min": round(mn, 6),
                "ising_dynamic_range": round(mx / mn, 2),
                "ising_at_risk_percent": round(100 * risk, 2)}

    def _formulate_qubo(self, net):
        self._prepare_network(net)
        self._encode_variables(net)
        self._resolve_penalties(net)

        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
        bqm = self._build_objective(bqm, net)
        if self.formulation == "dc_ptdf":
            bqm = self._build_power_balance_ptdf(bqm, net)
            bqm = self._build_line_limits_ptdf(bqm, net)
        else:
            bqm = self._build_power_balance_theta(bqm, net)
            bqm = self._build_line_limits_theta(bqm, net)

        R = self.var_registry
        num_dispatch_qubits = sum(len(reg['bits']) for et in self._DISPATCH_TYPES for reg in R[et].values())
        num_slack_qubits = sum(len(r['bits']) for r in R['slack_lines'].values())
        num_angle_qubits = sum(len(r['bits']) for r in R['bus'].values())
        n_monitored = sum(1 for ln in self._lines if ln['p_max'] > 0.0)

        n_vars = len(bqm.variables)
        max_possible_interactions = (n_vars * (n_vars - 1)) / 2
        density = (bqm.num_interactions / max_possible_interactions) if max_possible_interactions > 0 else 0.0
        
        mags = [abs(v) for v in bqm.linear.values() if abs(v) > 1e-12] + \
               [abs(v) for v in bqm.quadratic.values() if abs(v) > 1e-12]
        max_mag = max(mags) if mags else 0.0
        min_mag = min(mags) if mags else 0.0
        dynamic_range = (max_mag / min_mag) if min_mag > 0 else 1.0

        complexity = {
            "formulation": self.formulation,
            "encoding": self.encoding,
            "classical_domain_input": {
                "continuous_variables": sum(len(R[et]) for et in self._DISPATCH_TYPES)
                                        + sum(1 for r in R['bus'].values() if r['bits']),
                "equality_constraints": 1 if self.formulation == "dc_ptdf" else len(self._active_buses),
                "inequality_constraints": 2 * n_monitored,
            },
            "quantum_domain_qubo": {
                "total_logical_qubits": len(bqm.variables),
                "qubits_used_for_variables": len(bqm.variables) - num_slack_qubits,
                "qubits_used_for_dispatch": num_dispatch_qubits,
                "qubits_used_for_angles": num_angle_qubits,
                "qubits_wasted_on_slack": num_slack_qubits,
                "num_interactions": bqm.num_interactions,
                "density_percent": round(density * 100, 2),
                "offset": float(bqm.offset),
            },
            "hardware_limits": self._range_metrics(bqm),
            "penalties": {
                "balance": float(self.lambda_balance), 
                "line": self.lambda_line if isinstance(self.lambda_line, dict) else float(self.lambda_line)
            },
        }

        # Override legacy limits with new accurate hardware limit data
        complexity["hardware_limits"]["max_coefficient_magnitude"] = round(float(max_mag), 4)
        complexity["hardware_limits"]["min_coefficient_magnitude"] = round(float(min_mag), 4)
        complexity["hardware_limits"]["dac_precision_warning"] = bool(dynamic_range > 256.0)

        return bqm, complexity

    def _encoded_nodal_mismatch(self, sample, net, raw):
        ang = {b: self._decode_value(reg, sample) for b, reg in self.var_registry['bus'].items()}
        mis = {b: 0.0 for b in self._active_buses}
        for et, idx, reg, bus in self._dispatch_assets(net):
            mis[bus] += raw[et][idx]
        for bus, load in self._load_mw.items():
            mis[bus] -= load
        for ln in self._lines:
            flow = ln['b'] * (ang[ln['f']] - ang[ln['t']])
            mis[ln['f']] -= flow
            mis[ln['t']] += flow
        return max((abs(m) for m in mis.values()), default=0.0)

    def _decode_solution(self, sample, net):
        # Swap in the full, untruncated PTDF for honest physical evaluation
        truncated = getattr(self, '_H', {})
        if getattr(self, 'decode_with_full_ptdf', False) and hasattr(self, '_H_full'):
            self._H = self._H_full
            
        try:
            R = self.var_registry
            raw = {et: {idx: self._decode_value(reg, sample) for idx, reg in R[et].items()} for et in self._DISPATCH_TYPES}
            pos = self._bus_pos

            # --- physical state of the decoded dispatch ---
            inj = np.zeros(len(pos))
            bounds_ok = True
            for et, idx, reg, bus in self._dispatch_assets(net):
                val = raw[et][idx]
                if val < reg['min'] - 1e-6 or val > reg['max'] + 1e-6:
                    bounds_ok = False
                inj[pos[bus]] += val
            for bus, load in self._load_mw.items():
                inj[pos[bus]] -= load
            imbalance = float(inj.sum())                                      
            flows = {idx: float(h @ inj) for idx, h in self._H.items()}   
            angles = self._X @ inj

            max_line_violation = max([abs(flows[ln['idx']]) - ln['p_max'] for ln in self._lines if ln['p_max'] > 0.0] + [0.0])

            # --- operating point that is reported ---
            reported = {et: dict(v) for et, v in raw.items()}
            raw_ref = raw['ext_grid'][self._ref_ext]
            rebalance_ok = True
            if self.rebalance_slack:
                new_ref = raw_ref - imbalance
                reported['ext_grid'][self._ref_ext] = new_ref
                ref_reg = R['ext_grid'][self._ref_ext]
                rebalance_ok = ref_reg['min'] - 1e-6 <= new_ref <= ref_reg['max'] + 1e-6

            tol = self.feasibility_tol_mw if self.feasibility_tol_mw is not None else self.mw_precision
            balance_ok = abs(imbalance) <= tol
            lines_ok = (max_line_violation <= tol)

            cost = self._total_true_cost(reported)
            feasibility = {
                "total_generation_mw": round(float(sum(sum(v.values()) for v in reported.values())), 3),
                "total_load_mw": round(float(sum(self._load_mw.values())), 3),
                "raw_imbalance_mw": round(imbalance, 3),
                "raw_slack_dispatch_mw": round(float(raw_ref), 3),
                "cost_raw_dispatch_eur": round(self._total_true_cost(raw), 3),
                "max_line_violation_mw": round(float(max_line_violation), 3),
                "tolerance_mw": tol,
                "balance_ok": bool(balance_ok),
                "lines_ok": bool(lines_ok),
                "bounds_ok": bool(bounds_ok and rebalance_ok),
                "is_feasible": bool(balance_ok and lines_ok and bounds_ok and rebalance_ok),
                "line_flows_mw": {k: round(v, 3) for k, v in flows.items()},
                "bus_angles_rad": {b: round(float(angles[i]), 6) for b, i in pos.items()},
            }
            if self.formulation == "dc_theta":
                feasibility["max_nodal_mismatch_mw"] = round(float(self._encoded_nodal_mismatch(sample, net, raw)), 3)

            dispatch = {i: {'p_mw': round(v, 3)} for i, v in reported['gen'].items()}
            sgen_dispatch = {i: {'p_mw': round(v, 3)} for i, v in reported['sgen'].items()}
            slack_dispatch = {i: {'p_mw': round(v, 3)} for i, v in reported['ext_grid'].items()}
            
            return dispatch, sgen_dispatch, slack_dispatch, round(cost, 3), feasibility
            
        finally:
            self._H = truncated

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

    @staticmethod
    def _variable_symbol(group, idx):
        return {"gen": f"P_gen{idx}", "sgen": f"P_sgen{idx}", "ext_grid": f"P_ext{idx}",
                "bus": f"theta_{idx}"}[group]

    def _print_problem_parameters(self, net, complexity):
        line = "=" * 80
        ptdf = self.formulation == "dc_ptdf"
        print("\n" + line)
        print(f"   QUBO {self.formulation.upper()}-OPF: COMPLETE PROBLEM DEFINITION")
        print(line)
        print(f"Total Grid Demand (Load): {round(sum(self._load_mw.values()), 3)} MW")
        print(f"MW precision: {self.mw_precision} MW | Encoding: {self.encoding} | Reference bus: {self._ref_bus}\n")

        labels = {'gen': 'Gen', 'sgen': 'SGen', 'ext_grid': 'Ext_Grid (Slack)'}

        # 1) objective ------------------------------------------------------
        print("--- OBJECTIVE: minimise total generation cost ---")
        for et in self._DISPATCH_TYPES:
            tbl = getattr(net, et)
            for idx in self.var_registry[et]:
                c0, c1, c2 = self._cost_coeffs(et, idx)
                sym = self._variable_symbol(et, idx)
                entry = self._costs[et].get(int(idx))
                note = "   [fallback cost: none given in net]" if entry is None else (
                    "   [pwl_cost -> quadratic fit in QUBO]" if entry['pwl'] is not None else "")
                print(f" {labels[et]} {idx} (Bus {tbl.at[idx, 'bus']}): "
                      f"{c2:.6g}*{sym}^2 + {c1:.6g}*{sym} + {c0:.6g}{note}")
        for w in self.cost_model_warnings:
            print(f" [cost warning] {w}")
        print(" QUBO energy = sum(costs) + lambda_balance * (balance residual)^2"
              + (" + lambda_line * sum_lines(line residual)^2"))
        print(" Reported cost = exact cost function of the physically balanced operating point.")

        # 2) variables ------------------------------------------------------
        print(f"\n--- DECISION VARIABLES ({self.encoding} encoding: value = min + sum_k w_k * bit_k) ---")
        for et in self._DISPATCH_TYPES:
            for idx, reg in self.var_registry[et].items():
                sym = self._variable_symbol(et, idx)
                print(f" {sym:<10} in [{reg['min']}, {reg['max']}] MW | {len(reg['bits']):>4} bits | "
                      f"weights: {self._fmt_weights(reg['weights'])}")
        for bus, reg in self.var_registry['bus'].items():
            sym = self._variable_symbol('bus', bus)
            if reg['bits']:
                print(f" {sym:<10} in [{reg['min']:.4f}, {reg['max']:.4f}] rad | {len(reg['bits']):>4} bits | "
                      f"step {reg['precision']:.3g} rad")
            else:
                print(f" {sym:<10} fixed at 0 (slack reference or no lines)")
        for l_idx, reg in self.var_registry['slack_lines'].items():
            print(f" s_line{l_idx:<4} in [0, {reg['max']:.4g}] MW | {len(reg['bits']):>4} bits | "
                  f"(inequality slack, not a physical variable)")
        if ptdf:
            print(" (no angle variables: angles are recovered afterwards as theta = X @ injections)")

        # 3) network and constraints ---------------------------------------
        if ptdf:
            print(f"\n--- TRANSMISSION LINES (PTDF: flow_l = sum_i PTDF[l,i] * (P_i - load_i), bus {self._ref_bus} = reference) ---")
        else:
            print("\n--- TRANSMISSION LINES (flow_ij = B_ij * (theta_i - theta_j)) ---")
        inv_pos = {i: b for b, i in self._bus_pos.items()}
        for ln in self._lines:
            limit_str = f"|flow| <= {round(ln['p_max'], 2)} MW" if ln['p_max'] > 0 else "unconstrained"
            print(f" Line {ln['idx']} ({ln['f']} -> {ln['t']}): B = {round(ln['b'], 2)} MW/rad, {limit_str}")
            if ptdf and ln['p_max'] > 0:
                row = ", ".join(f"bus{inv_pos[i]} {v:+.4f}" for i, v in enumerate(self._H[ln['idx']]) if abs(v) > 1e-9)
                print(f"     PTDF: {row}")

        print("\n--- CONSTRAINTS (soft, as quadratic penalties) ---")
        if ptdf:
            syms = " + ".join(self._variable_symbol(et, idx) for et in self._DISPATCH_TYPES for idx in self.var_registry[et])
            print(f" Balance: {syms} - {round(sum(self._load_mw.values()), 3)} MW load = 0")
        else:
            for bus in sorted(self._active_buses):
                inj = [self._variable_symbol(et, idx) for et, idx, reg, b in self._dispatch_assets(net) if b == bus]
                flows = []
                for ln in self._lines:
                    if ln['f'] == bus:
                        flows.append(f"flow({ln['f']}->{ln['t']})")
                    elif ln['t'] == bus:
                        flows.append(f"flow({ln['t']}->{ln['f']})")
                load = round(self._load_mw.get(bus, 0.0), 3)
                print(f" KCL bus {bus}: {' + '.join(inj) if inj else '0'} - {load} MW load "
                      f"- outgoing[{', '.join(flows)}] = 0")
        for l_idx in self.var_registry['slack_lines']:
            ln = next(x for x in self._lines if x['idx'] == l_idx)
            print(f" Line {l_idx}: {round(ln['p_max'], 2)} + flow({ln['f']}->{ln['t']}) - s_line{l_idx} = 0")

        auto_b = "auto" if self.penalty_balance is None else "manual"
        auto_l = "auto" if self.penalty_line is None else "manual"
        
        if isinstance(self.lambda_line, dict):
            l_min = min(self.lambda_line.values()) if self.lambda_line else 0.0
            l_max = max(self.lambda_line.values()) if self.lambda_line else 0.0
            line_str = f"[{l_min:.2f} to {l_max:.2f}]"
        else:
            line_str = f"{self.lambda_line:.6g}"
            
        print(f" lambda_balance = {self.lambda_balance:.6g} ({auto_b}) | lambda_line = {line_str} ({auto_l})")

        # 4) size summary ---------------------------------------------------
        c, q = complexity['classical_domain_input'], complexity['quantum_domain_qubo']
        print("\n--- PROBLEM SIZE ---")
        print(f" Classical: {c['continuous_variables']} continuous variables, "
              f"{c['equality_constraints']} equality and {c['inequality_constraints']} inequality constraints")
        print(f" QUBO ({self.encoding}): {q['total_logical_qubits']} logical qubits "
              f"(dispatch {q['qubits_used_for_dispatch']}, angles {q['qubits_used_for_angles']}, "
              f"line-limit slack {q['qubits_wasted_on_slack']}), "
              f"{q['num_interactions']} quadratic terms, constant offset {q['offset']:.6g}")
        print(line + "\n")


if __name__ == "__main__":
    import copy
    import warnings
    import itertools
    import pandas as pd

    import pandapower as pp
    import pandapower.networks as pn
    from homemade_grids.small_grids import case3_low_gen

    warnings.filterwarnings("ignore")
    net = case3_low_gen()

    print("\n" + "="*115)
    print(" QUBO CONSTANT EXPLORATION (case5) ".center(115, "="))
    print("="*115)
    
    rows = []
    
    # Iterate through all 8 combinations
    combinations = itertools.product(
        QuboFormulator.FORMULATIONS, 
        QuboFormulator.ENCODINGS, 
        [10.0, 1.0]
    )
    
    for form, enc, prec in combinations:
        f = QuboFormulator(formulation=form, encoding=enc, mw_precision=prec)
        bqm, cx = f._formulate_qubo(net)
        
        # Extract constants
        h_mags = [abs(v) for v in bqm.linear.values() if abs(v) > 1e-12]
        j_mags = [abs(v) for v in bqm.quadratic.values() if abs(v) > 1e-12]
        
        max_h = max(h_mags) if h_mags else 0.0
        max_j = max(j_mags) if j_mags else 0.0
        
        q = cx["quantum_domain_qubo"]
        hw = cx["hardware_limits"]
        pens = cx["penalties"]
        
        rows.append({
            "Form": form,
            "Enc": enc,
            "Prec": prec,
            "Qubits": q["total_logical_qubits"],
            "Penalty (\u03BB)": round(pens["balance"], 1),
            "Offset": round(bqm.offset, 1),
            "Max Lin (h)": round(max_h, 1),
            "Max Quad (J)": round(max_j, 1),
            "Dyn Range": round(hw["dynamic_range"], 1)
        })
        
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("="*115 + "\n")
    
    # --- DEEP DIVE: The Epiphany ---
    print("--- DEEP DIVE: WHY RADIX EXPLODES (dc_ptdf, radix, prec=1.0) ---")
    f = QuboFormulator(formulation="dc_ptdf", encoding="radix", mw_precision=1.0)
    bqm, cx = f._formulate_qubo(net)
    
    # Find the single largest interaction in the matrix
    max_j_val = 0
    max_j_pair = None
    for (u, v), val in bqm.quadratic.items():
        if abs(val) > max_j_val:
            max_j_val = abs(val)
            max_j_pair = (u, v)
            
    pen = cx['penalties']['balance']
    print(f"Largest Matrix Term (J): {max_j_val:,.1f}")
    print(f"Occurs between variables : {max_j_pair[0]}  AND  {max_j_pair[1]}")
    print("\nTHE EPIPHANY:")
    print("In Radix encoding, the highest bits carry massive weight (e.g., 128 MW or 256 MW).")
    print(f"When the power balance equation squares the sum, it multiplies those bits together:")
    print(f" (Bit_A * Bit_B) * 2 * Penalty")
    print(f" (256 MW * 256 MW) * 2 * {pen:.1f} = ~{256 * 256 * 2 * pen:,.0f}!")
    print("\nCompare this to Unary encoding, where every bit is exactly 1.0 MW:")
    print(f" (1.0 MW * 1.0 MW) * 2 * {pen:.1f} = ~{1 * 1 * 2 * pen:,.0f}")
    print("This is why Unary encoding collapses the dynamic range.\n")

    # --- QUBO MATRIX VISUALIZATION ---
    print("="*115)
    print(" VISUAL QUBO MATRIX (dc_ptdf, radix, prec=10.0) ".center(115, "="))
    print("="*115)
    print("Diagonal = Linear biases (h). Upper Triangle = Quadratic interactions (J).")
    print("Empty cells = 0.0 (No interaction). Values are rounded to nearest whole number.\n")

    f_viz = QuboFormulator(formulation="dc_ptdf", encoding="radix", mw_precision=10.0)
    bqm_viz, _ = f_viz._formulate_qubo(net)

    # Abbreviate names so the Pandas table fits on screen
    def short_name(name):
        name = name.replace("gen_", "g")
        name = name.replace("ext_", "e")
        name = name.replace("slack_line_", "sl")
        name = name.replace("_bit_", "_b")
        return name

    variables = list(bqm_viz.variables)
    variables.sort() # Sorts alphabetically (ext, gen, slack) for grouped viewing
    
    short_vars = [short_name(v) for v in variables]
    matrix = pd.DataFrame(index=short_vars, columns=short_vars)
    matrix = matrix.fillna("") # Fill with empty strings for visual sparsity

    for i, v1 in enumerate(variables):
        # Linear terms go on the diagonal
        val_lin = bqm_viz.linear.get(v1, 0.0)
        if abs(val_lin) > 1e-3:
            matrix.iloc[i, i] = f"{val_lin:,.0f}"

        # Quadratic terms go in the upper triangle
        for j in range(i + 1, len(variables)):
            v2 = variables[j]
            val_quad = bqm_viz.quadratic.get((v1, v2), bqm_viz.quadratic.get((v2, v1), 0.0))
            if abs(val_quad) > 1e-3:
                matrix.iloc[i, j] = f"{val_quad:,.0f}"

    # Force pandas to print the whole table without truncating columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 2000)
    
    print(matrix)
    print("="*115 + "\n")