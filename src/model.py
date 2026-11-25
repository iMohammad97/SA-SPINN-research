"""Coupled separable PINN: SOH branch (per cycle) + SOC branch (per second).

One physics template for both states:  Δû/Δτ = prior + F_φ(q)
  SOH: prior = 0 (learned degradation rate)          τ = cycles/100
  SOC: prior = ΔQ / (SOH_hat · Q_NOM)  (Coulomb)     τ = hours
Coupling SOH -> SOC: SOC prior uses SOH_hat; SOC net also sees SOH_hat.
Coupling SOC -> SOH (explicit, gamma > 0): long-span capacity consistency within a cycle
  (u_b - u_a) * SOH_hat * Q_NOM = Q_b - Q_a      [Ah]
"""
import torch
import torch.nn as nn
from data import HI, GROUPS

Q_NOM = 2.0   # default; train.py sets model.q_nom per dataset batch (ds.Q_NOM)
_c = {n: i for i, n in enumerate(HI)}
SOH_GROUPS = {
    "sensor": [[_c[n] for n in GROUPS[g]] + ([_c["cycle"]] if g == "clock" else []) for g in GROUPS],
    "statistic": [[_c[n] for n in g] for g in (
        ["v_mean", "i_mean"], ["v_std", "i_std", "t_std"], ["v_kurt", "i_kurt"], ["v_skew", "i_skew"],
        ["v_slope", "i_slope", "t_rise"], ["v_entropy", "i_entropy"], ["cc_q", "cv_q"],
        ["cc_time", "cv_time", "cycle"])],
    "all_in_one": [list(range(len(HI)))],
}
# SOC inputs: V, I, T, t, SOH_hat (index 4)
SOC_GROUPS = {"sensor": [[0], [1], [2], [3, 4]], "statistic": [[0, 1], [2, 3, 4]], "all_in_one": [[0, 1, 2, 3, 4]]}


def tdrop(xs, cols, flat, p):
    """Temperature dropout on rows of xs (same draw for each tensor in xs): share p see a flat sensor
    (raw value 0 -> `flat` after scaling), share p a missing one (training mean -> 0)."""
    r = torch.rand(len(xs[0]), 1)
    out = []
    for x in xs:
        x = x.clone()
        x[:, cols] = torch.where(r < p, flat, torch.where(r < 2 * p, torch.zeros_like(flat), x[:, cols]))
        out.append(x)
    return out


def mlp(i, h, o, n=3):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class Fusion(nn.Module):
    """Tokens [B, n, p] -> scalar. mean / owa / choquet / attention / product."""

    def __init__(self, kind, n, p):
        super().__init__()
        self.kind = kind
        self.value = nn.Linear(p, 1)
        self.w = nn.Parameter(torch.zeros(n))
        if kind == "log_uaw":                             # learned logits + log-precision prior
            self.wl = nn.Parameter(torch.zeros(n, p))
        if kind.endswith("attention"):
            self.attn = nn.MultiheadAttention(p, 2, batch_first=True)
        self.out = nn.Linear(1, 1)
        self.register_buffer("prec", torch.ones(n, p), persistent=False)   # 1/var of log-tokens under the budget, ivar/ci

    def forward(self, z, var=None):
        if self.kind == "product":                        # SPINN canonical: sum_r prod_j z_jr
            return self.out(self.value(z.prod(1))).squeeze(1)
        if self.kind.startswith("log_"):                  # product = exp(n * Agg_j log z_j)
            return self.out(self.value(self._log_fuse(z, self.kind[4:], var))).squeeze(1)
        if self.kind == "attention":
            z = self._attend(z)
        v = self.value(z).squeeze(-1)                     # [B, n]
        if self.kind in ("mean", "attention"):
            f = v.mean(1)
        else:
            vs = v.sort(1, descending=True)[0]
            w = torch.softmax(self.w, 0)
            if self.kind == "owa":
                f = (vs * w).sum(1)
            else:                                         # choquet, cardinality capacity
                mu = torch.cumsum(w, 0)
                nxt = torch.cat([vs[:, 1:], torch.zeros_like(vs[:, :1])], 1)
                f = ((vs - nxt) * mu).sum(1)
        return self.out(f.unsqueeze(1)).squeeze(1)

    def _attend(self, z):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):                # math kernel: supports double backward (C6a)
            return self.attn(z, z, z, need_weights=False)[0]

    def _log_fuse(self, z, agg, var=None):
        """Log-domain fusion per rank component r: G_r = exp(n * Agg_j log z_jr).
        mean -> product; owa/choquet/trim/wmean/ivar/ci -> robust / weighted products."""
        if agg == "attention":
            z = self._attend(z)
        l = torch.log(nn.functional.softplus(z) + 1e-4)   # [B, n, p], positive tokens
        n = l.shape[1]
        w = torch.softmax(self.w, 0)[None, :, None]
        if agg in ("mean", "attention"):
            g = l.mean(1)
        elif agg == "wmean":                              # fixed-weight DST (global masses)
            g = (w * l).sum(1)
        elif agg == "uaw":                                # w_jr ∝ exp(θ_jr) / var_jr (propagated, C6b)
            g = (torch.softmax(self.wl[None] - torch.log(var + 1e-4), 1) * l).sum(1)
        elif agg == "ivar":                               # inverse-variance (BLUE) weights per component
            g = (self.prec / self.prec.sum(0) * l).sum(1)
        elif agg == "ci":                                 # scalar covariance intersection -> least-uncertain token
            g = (nn.functional.one_hot(self.prec.argmax(0), n).T.float() * l).sum(1)
        elif agg == "trim":                               # drop min & max token
            g = l.sort(1)[0][:, 1:-1].mean(1) if n > 2 else l.mean(1)
        else:
            s = l.sort(1, descending=True)[0]
            if agg == "owa":
                g = (w * s).sum(1)
            else:                                         # choquet, cardinality capacity
                nxt = torch.cat([s[:, 1:], torch.zeros_like(s[:, :1])], 1)
                g = ((s - nxt) * torch.cumsum(w, 1)).sum(1)
        return torch.exp((n * g).clamp(max=20.0))


class Branch(nn.Module):
    def __init__(self, groups, fusion, h=16, p=8, depth=2):
        super().__init__()
        self.groups = groups
        self.subnets = nn.ModuleList(mlp(len(g), h, p, depth + 1) for g in groups)
        self.fusion = Fusion(fusion, len(groups), p)
        self.register_buffer("sigma", torch.zeros(1 + max(max(g) for g in groups)).diag(), persistent=False)

    def token_var(self, x):
        """First-order Var of log-tokens: J_j Σ_jj J_jᵀ; sensor-aligned -> each token sees its own Σ block."""
        from torch.func import jacfwd, vmap
        out = []
        for s, g in zip(self.subnets, self.groups):
            f = lambda xi, s=s: torch.log(nn.functional.softplus(s(xi)) + 1e-4)
            J = vmap(jacfwd(f))(x[:, g].detach())                    # [B, p, |g|]
            out.append(torch.einsum("bpi,ij,bpj->bp", J, self.sigma[g][:, g], J))
        return torch.stack(out, 1).detach()

    def forward(self, x):
        z = torch.stack([s(x[:, g]) for s, g in zip(self.subnets, self.groups)], 1)
        return self.fusion(z, self.token_var(x) if self.fusion.kind == "log_uaw" else None)


class CoupledSPINN(nn.Module):
    def __init__(self, grouping="sensor", fusion="mean", physics="gray", couple="two", h=16, depth=2):
        super().__init__()
        self.physics, self.couple, self.q_nom = physics, couple, Q_NOM  # couple: two | oneway (SOH->SOC, no grad back) | none
        self.soh = Branch(SOH_GROUPS[grouping], fusion, h, depth=depth)    # depth = hidden layers per sub-network
        self.soc = Branch(SOC_GROUPS[grouping], fusion, h, depth=depth)
        self.F_soh = mlp(len(HI) + 1, 16, 1)
        self.F_soc = mlp(6, 16, 1)

    def cap(self, s):
        """SOH seen by the SOC side: estimate (two-way), detached (one-way cascade) or nominal (decoupled)."""
        return {"two": s, "oneway": s.detach(), "none": torch.ones_like(s)}[self.couple]

    def soc_in(self, z, s):
        return torch.cat([z, self.cap(s).unsqueeze(1)], 1)

    def set_sigma(self, d):
        self.soh.sigma, self.soc.sigma = d["sx"].float(), d["sz"].float()

    def jsj(self, branch, x, y):
        """Uncertainty penalty: tr(J Σ Jᵀ) of output y w.r.t. branch input x (graph kept)."""
        J = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        return ((J @ branch.sigma) * J).sum(1).mean()

    def losses(self, d, split, bs_soc, bs_soh, alpha, beta, gamma=0.0, lam_f=0.0, pen=0.0, aug=False, td=0.0):
        D, X, Z = d[split], d["X"], d["Z"]
        # --- SOH pairs
        j = torch.randint(len(D["soh_i"]), (bs_soh,))
        i1 = D["soh_i"][j]
        x1, x2 = X[i1], X[i1 + 1]
        if aug:                                           # augmentation: one budget draw per pair (persistent)
            k = torch.randint(len(d["draws"]), (bs_soh,))
            x1, x2 = d["draws"][k, i1], d["draws"][k, i1 + 1]
        if td:                                            # P4: flat or missing temperature sensor
            tc = [_c["t_std"], _c["t_rise"]]
            xf = (-d["xstat"][0][tc] / d["xstat"][1][tc]).float()
            zf = torch.tensor(-d["zstat"][0][2:3] / d["zstat"][1][2:3], dtype=torch.float32)
            x1, x2 = tdrop([x1, x2], tc, xf, td)
        if pen:
            x1 = x1.clone().requires_grad_(True)
        s1, s2 = self.soh(x1), self.soh(x2)
        y1, y2 = d[split]["soh_y"][i1], d[split]["soh_y"][i1 + 1]
        if "soh_lab" in D:                                # SOH-scarce (RPT): labels on a few cycles only
            rs = D["soh_lab"][torch.randint(len(D["soh_lab"]), (bs_soh,))]
            l_soh = ((self.soh(X[rs]) - D["soh_y"][rs]) ** 2).mean()
            mono = torch.relu(s2 - s1).mean()             # no labels -> SOH non-increasing
        else:
            l_soh = 0.5 * ((s1 - y1) ** 2 + (s2 - y2) ** 2).mean()
            mono = torch.relu((s2 - s1) * (y1 - y2)).mean()
        r_soh = (s2 - s1) / D["dk"][j] - self.F_soh(torch.cat([x1, s1[:, None]], 1)).squeeze(1)
        # --- SOC pairs (coupled through SOH_hat of the same cycle)
        k = torch.randint(len(D["soc_i"]), (bs_soc,))
        r1 = D["soc_i"][k]
        xh = X[d["hi_idx"][r1]]
        z1, z2 = Z[r1], Z[r1 + 1]
        if aug:
            xh = d["draws"][torch.randint(len(d["draws"]), (bs_soc,)), d["hi_idx"][r1]]
            dz = torch.randn(bs_soc, Z.shape[1]) * d["sz"][:Z.shape[1]].diag().sqrt().float()
            z1, z2 = z1 + dz, z2 + dz
        if td:
            xh, = tdrop([xh], tc, xf, td)
            z1, z2 = tdrop([z1, z2], [2], zf, td)
        sh = self.soh(xh)
        zin = self.soc_in(z1, sh)
        if pen and not zin.requires_grad:                 # decoupled baselines: input is a leaf
            zin.requires_grad_(True)
        u1, u2 = self.soc(zin), self.soc(self.soc_in(z2, sh))
        if "lab" in D:                                    # label-scarce: data loss on labelled rows only
            rl = D["lab"][torch.randint(len(D["lab"]), (bs_soc,))]
            ul = self.soc(self.soc_in(Z[rl], self.soh(X[d["hi_idx"][rl]])))
            l_soc = ((ul - d["soc_y"][rl]) ** 2).mean()
        else:
            l_soc = 0.5 * ((u1 - d["soc_y"][r1]) ** 2 + (u2 - d["soc_y"][r1 + 1]) ** 2).mean()
        rate = (u2 - u1) / D["dt_h"][k]
        prior = D["dq"][k] / (self.cap(sh).clamp_min(0.3) * self.q_nom) / D["dt_h"][k]
        corr = self.F_soc(torch.cat([z1, u1[:, None], sh[:, None]], 1)).squeeze(1)
        r_soc = rate - {"cc": prior, "gray": prior + corr, "learned": corr}[self.physics]
        loss = l_soh + l_soc + alpha * ((r_soh ** 2).mean() + (r_soc ** 2).mean()) + beta * mono
        if lam_f > 0 and self.physics == "gray":        # keep correction small -> prior stays identifiable
            loss = loss + lam_f * (corr ** 2).mean()
        parts = dict(soh=l_soh.item(), soc=l_soc.item(), r_soh=(r_soh ** 2).mean().item(),
                     r_soc=(r_soc ** 2).mean().item())
        if pen:
            p_soh, p_soc = self.jsj(self.soh, x1, s1), self.jsj(self.soc, zin, u1)
            loss = loss + pen * (p_soh + p_soc)
            parts["pen"] = (p_soh + p_soc).item()
        if gamma > 0:                                     # explicit SOC -> SOH consistency
            rb = d["seg0"][r1] + (torch.rand(bs_soc) * d["segn"][r1]).long()
            ub = self.soc(self.soc_in(Z[rb], sh))
            r_cons = (ub - u1) * sh * self.q_nom - (d["q"][rb] - d["q"][r1]).float()
            loss = loss + gamma * (r_cons ** 2).mean()
            parts["r_cons"] = (r_cons ** 2).mean().item()
        return loss, parts

    @torch.no_grad()
    def evaluate(self, d, split):
        D, X = d[split], d["X"]
        rows = D["soh_rows"]
        s = self.soh(X[rows])
        y, a = D["soh_y"][rows], D["anchor"][rows]
        soh_rmse = ((s - y)[a] ** 2).mean().sqrt().item()          # measured (anchor) cycles only
        r = D["soc_rows"]
        sh = self.soh(X[d["hi_idx"][r]])
        u = torch.cat([self.soc(self.soc_in(d["Z"][r[i:i + 65536]], sh[i:i + 65536]))
                       for i in range(0, len(r), 65536)])
        soc_rmse = ((u - d["soc_y"][r]) ** 2).mean().sqrt().item()
        return soh_rmse, soc_rmse
