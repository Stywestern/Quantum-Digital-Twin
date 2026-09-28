"""ScaledQuboFormulator: drop-in subclass of your QuboFormulator that shrinks the QUBO dynamic range.

Your original file is untouched. Every change is a constructor flag, so each one can be ablated:

  scale_line_constraints   (dc_ptdf) each line-limit constraint is divided by its largest PTDF entry sigma, so the
                           constraint is written in "equivalent generator MW". Removes the 1/H^2 spread inside one
                           constraint AND fixes lambda_line: a violation e (scaled) is worth cost ~ marginal*e, so
                           lambda_line = safety * (max marginal cost of the gens that can move this line) / precision.
  smart_slack_side         (dc_ptdf) use a one-sided slack range when only one side of |flow|<=p_max can be violated
                           (smaller range -> fewer bits / smaller weights). Two-sided case is unchanged.
  slack_precision_factor   slack step = factor * mw_precision (in scaled units). Keep <= ~2: a coarser slack leaves a
                           rounding residual that is itself penalised, which distorts feasible states.
  decode_with_full_ptdf    the decoder judges line flows with the UNTRUNCATED PTDF (physical truth) instead of the
                           truncated one used inside the QUBO. Honest, but may reveal violations the QUBO cannot see.
  ptdf_rel_threshold       also zero PTDF entries below rel * max|H_l| of that line (None = off).
  ptdf_round_to            round the (kept) PTDF entries to a multiple of this value (None = off).
  snap_weights             round encoding weights down to multiples of the precision (kills odd tiny leftovers).
  ext_fallback_mult        multiplier on the fallback cost of an ext_grid that has no cost entry (base file: 10).
  risk_noise_frac          fraction of the max Ising coefficient regarded as the hardware noise floor.

Also adds Ising-form range metrics to complexity['hardware_limits'] and coefficient_report().
"""
import math

import dimod
import numpy as np

from solvers.qubo_formulator import QuboFormulator as _Base       # <- your file


def _cat(v):
    s = str(v)
    if s.startswith("slack_line_"):
        return "slack"
    if s.startswith("bus_"):
        return "angle"
    return "disp"


class ScaledQuboFormulator(_Base):
    def __init__(self, *args, scale_line_constraints=True, smart_slack_side=True,
                 slack_precision_factor=1.0, decode_with_full_ptdf=True,
                 ptdf_rel_threshold=None, ptdf_round_to=None, snap_weights=True,
                 ext_fallback_mult=10.0, risk_noise_frac=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_line_constraints = scale_line_constraints
        self.smart_slack_side = smart_slack_side
        self.slack_precision_factor = slack_precision_factor
        self.decode_with_full_ptdf = decode_with_full_ptdf
        self.ptdf_rel_threshold = ptdf_rel_threshold
        self.ptdf_round_to = ptdf_round_to
        self.snap_weights = snap_weights
        self.ext_fallback_mult = ext_fallback_mult
        self.risk_noise_frac = risk_noise_frac

    # ------------------------------------------------------------------ costs
    def _cost_coeffs(self, et, idx):
        entry = self._costs[et].get(int(idx))
        if entry is not None:
            return entry['coeffs']
        return (0.0, self._fallback_c1 * (self.ext_fallback_mult if et == 'ext_grid' else 1.0), 0.0)

    # --------------------------------------------------------------- encoding
    def _get_weights(self, total_range, precision):
        w = super()._get_weights(total_range, precision)
        if not self.snap_weights or self.encoding == "iterative" or not precision or precision <= 0:
            return w
        snapped = []
        for x in w:
            k = math.floor(x / precision + 1e-9)
            if k > 0:
                snapped.append(k * precision)
        # never make the variable unrepresentable (range smaller than one step): keep the original then
        return snapped if snapped else w

    # ------------------------------------------------------------------- PTDF
    def _build_ptdf(self, net):
        super()._build_ptdf(net)                       # X, bus positions, truncated self._H
        pos, X = self._bus_pos, self._X
        self._H_full = {ln['idx']: ln['b'] * (X[pos[ln['f']]] - X[pos[ln['t']]]) for ln in self._lines}
        if self.ptdf_rel_threshold is None and self.ptdf_round_to is None:
            return
        for ln in self._lines:
            raw = self._H_full[ln['idx']]
            thr = max(self.ptdf_threshold, (self.ptdf_rel_threshold or 0.0) * float(np.max(np.abs(raw))))
            h = np.where(np.abs(raw) < thr, 0.0, raw)
            if self.ptdf_round_to:
                h = np.round(h / self.ptdf_round_to) * self.ptdf_round_to
            self._H[ln['idx']] = h

    def _decode_solution(self, sample, net):
        if not self.decode_with_full_ptdf:
            return super()._decode_solution(sample, net)
        truncated, self._H = self._H, self._H_full
        try:
            return super()._decode_solution(sample, net)
        finally:
            self._H = truncated

    # ------------------------------------------------------------ line limits
    def _build_line_limits_ptdf(self, bqm, net):
        if not self.scale_line_constraints:
            return super()._build_line_limits_ptdf(bqm, net)

        pos = self._bus_pos
        assets = list(self._dispatch_assets(net))
        slack_prec = self.mw_precision * self.slack_precision_factor

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
                    _, c1, c2 = self._cost_coeffs(et, idx)
                    marg.append(abs(c1) + 2.0 * abs(c2) * reg['max'])

            upper = max_f > p_max + 1e-9            # flow > +p_max reachable
            lower = min_f < -p_max - 1e-9           # flow < -p_max reachable
            if not (upper or lower):
                continue                            # cannot be violated: no penalty needed
            if not coefs:
                self.cost_model_warnings.append(f"{ln['idx']}: limit can be violated but no variable moves it; skipped")
                continue

            # constraint  p_max + sgn*flow - s = 0,  s in [0, ub]
            if upper and lower or (lower and not upper):
                sgn = 1.0
                ub = 2.0 * p_max if (upper and lower) else p_max + max_f
            else:
                sgn = -1.0
                ub = p_max - min_f
            if not self.smart_slack_side:
                sgn, ub = 1.0, min(p_max + max_f, 2.0 * p_max)
            if ub <= 1e-9:
                self.cost_model_warnings.append(f"{ln['idx']}: limit violated in every state; skipped")
                continue

            sigma = max(coefs)                       # largest |PTDF| among movable variables
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

    # ---------------------------------------------------------------- metrics
    def _range_metrics(self, bqm):
        ising = bqm.change_vartype(dimod.SPIN, inplace=False)
        mags = [abs(b) for b in ising.linear.values() if abs(b) > 1e-12]
        mags += [abs(b) for _, _, b in ising.iter_quadratic() if abs(b) > 1e-12]
        if not mags:
            return {}
        mx, mn = max(mags), min(mags)
        risk = sum(1 for m in mags if m < self.risk_noise_frac * mx) / len(mags)
        return {"ising_max": round(mx, 4), "ising_min": round(mn, 6),
                "ising_dynamic_range": round(mx / mn, 2),
                "ising_at_risk_percent": round(100 * risk, 2)}

    def _formulate_qubo(self, net):
        bqm, cx = super()._formulate_qubo(net)
        cx['hardware_limits'].update(self._range_metrics(bqm))
        return bqm, cx

    def coefficient_report(self, bqm, print_table=True):
        """Ising-form coefficient magnitudes grouped by type (h / J, dispatch / slack / angle)."""
        ising = bqm.change_vartype(dimod.SPIN, inplace=False)
        groups = {}
        for v, b in ising.linear.items():
            groups.setdefault("h:" + _cat(v), []).append(abs(b))
        for u, v, b in ising.iter_quadratic():
            a, c = sorted((_cat(u), _cat(v)))
            groups.setdefault(f"J:{a}-{c}", []).append(abs(b))
        rep = {}
        for k, vals in groups.items():
            vals = [x for x in vals if x > 1e-12]
            if vals:
                rep[k] = {"count": len(vals), "max": max(vals), "min": min(vals), "median": float(np.median(vals))}
        if print_table:
            print(f"{'type':<16}{'count':>7}{'max':>14}{'median':>14}{'min':>14}")
            for k, r in sorted(rep.items(), key=lambda kv: -kv[1]['max']):
                print(f"{k:<16}{r['count']:>7}{r['max']:>14.4g}{r['median']:>14.4g}{r['min']:>14.4g}")
        return rep