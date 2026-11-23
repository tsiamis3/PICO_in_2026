"""
PICOSEC Micromegas — ιεραρχικό Bayesian μοντέλο για τα out-of-fold αποτελέσματα του timing network.

Ερώτηση: είναι η αβεβαιότητα σ_network του δικτύου σωστά βαθμονομημένη, και υπάρχει συστηματικό bias,
*τοπικά* — ανά charge και ανά θέση (hit_x, hit_y) — και όχι μόνο συνολικά;

Μοντέλο (ανά ανάλυση A1/A2/A3, όλα τα events, χωρίς pull cut):
    Δt_i        ~ StudentT(ν, μ_i, λ_i · σ_i)            Δt = t_network − t_true [ps], σ_i = σ_network [ps]
    μ_i         = μ0 + a[q_i] + b[c_i]                     bias: κοινό + charge bin + κελί θέσης
    log λ_i     = l0 + u[q_i] + v[c_i]                     κλίμακα αβεβαιότητας (λ = 1: σωστή βαθμονόμηση)
    a ~ N(0, τ_a),  b ~ N(0, τ_b),  u ~ N(0, τ_u),  v ~ N(0, τ_v)     ← partial pooling
    μ0 ~ N(0, 10 ps), l0 ~ N(0, 0.5), τ_a,τ_b ~ HalfNormal(5 ps), τ_u,τ_v ~ HalfNormal(0.3), ν−1 ~ Gamma(2, 0.1)
q_i: 20 bins ίσου πληθυσμού στο Q, c_i: κελιά 1.3 mm × 1.33 mm (20 × 12) στο (hit_x, hit_y).
Non-centered παραμετροποίηση, NUTS (NumPyro), 2 chains (vectorized) × 800 draws (800 warm-up).
Το posterior σώζεται στο data/hb_posterior/ — αν υπάρχει, τα plots ξαναβγαίνουν χωρίς νέο fit.

Τα κελιά με λίγα events (άκρα) «τραβιούνται» προς τον κοινό μέσο· τα γεμάτα κρατούν τη δική τους τιμή.

Τρέξιμο (από τον φάκελο του repo):
    pip install numpyro jax uproot pyarrow
    python scripts/hierarchical_bayes.py            (όλα)
    python scripts/hierarchical_bayes.py A2         (μόνο ένα fit· χρήσιμο αν η RAM είναι λίγη)
    python scripts/hierarchical_bayes.py A2 --draws 500   (λιγότερα draws)
Χρειάζεται τα ίδια αρχεία στο data/ με το picosec_analysis.py (Results + ROOT).
Βγάζει: plots/hb_*.png|pdf, plots_light/hb_*.png|pdf, plots/hierarchical_summary.json
"""
import os
import sys
import json
import gc

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as np_summary
from scipy import stats, optimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import picosec_analysis as pa                       # ίδια φόρτωση δεδομένων και ίδια θέματα με τα υπόλοιπα plots
from picosec_analysis import PicosecAnalysis, set_theme, THEME


# =====================================================================================
# Ομάδες: charge bins και κελιά θέσης
# =====================================================================================
class Grouping:
    """Αντιστοιχεί κάθε event σε charge bin (ίσου πληθυσμού) και σε κελί (hit_x, hit_y)."""
    def __init__(self, n_q=20, x_edges=np.linspace(-1.3, 1.3, 21), y_edges=np.linspace(-0.8, 0.8, 13)):
        self.n_q, self.x_edges, self.y_edges = n_q, np.asarray(x_edges), np.asarray(y_edges)
        self.nx, self.ny = len(self.x_edges) - 1, len(self.y_edges) - 1

    def __call__(self, D):
        self.q_edges = np.quantile(D["Q"].values, np.linspace(0, 1, self.n_q + 1))
        q = np.clip(np.searchsorted(self.q_edges, D["Q"].values, side="right") - 1, 0, self.n_q - 1)
        ix = np.clip(np.digitize(D["hit_x"].values, self.x_edges) - 1, 0, self.nx - 1)
        iy = np.clip(np.digitize(D["hit_y"].values, self.y_edges) - 1, 0, self.ny - 1)
        c = iy * self.nx + ix
        self.q_centre = np.array([D["Q"].values[q == j].mean() for j in range(self.n_q)])
        self.n_cell = np.bincount(c, minlength=self.nx * self.ny)
        return q.astype(np.int32), c.astype(np.int32)


# =====================================================================================
# Μοντέλο
# =====================================================================================
def hier_model(dt, sig, q, c, n_q, n_c):
    mu0 = numpyro.sample("mu0", dist.Normal(0.0, 10.0))
    l0 = numpyro.sample("l0", dist.Normal(0.0, 0.5))
    tau_a = numpyro.sample("tau_a", dist.HalfNormal(5.0))
    tau_b = numpyro.sample("tau_b", dist.HalfNormal(5.0))
    tau_u = numpyro.sample("tau_u", dist.HalfNormal(0.3))
    tau_v = numpyro.sample("tau_v", dist.HalfNormal(0.3))
    nu = 1.0 + numpyro.sample("nu_m1", dist.Gamma(2.0, 0.1))
    with numpyro.plate("qbins", n_q):
        za = numpyro.sample("za", dist.Normal(0, 1))
        zu = numpyro.sample("zu", dist.Normal(0, 1))
    with numpyro.plate("cells", n_c):
        zb = numpyro.sample("zb", dist.Normal(0, 1))
        zv = numpyro.sample("zv", dist.Normal(0, 1))
    # Χωρίς numpyro.deterministic: θα έκανε το get_samples() να ξανατρέχει όλο το μοντέλο (όλα τα events) σε κάθε draw
    a, b, u, v = tau_a * za, tau_b * zb, tau_u * zu, tau_v * zv
    mu = mu0 + a[q] + b[c]
    lam = jnp.exp(l0 + u[q] + v[c])
    with numpyro.plate("events", dt.shape[0]):
        numpyro.sample("dt", dist.StudentT(nu, mu, lam * sig), obs=dt)


def add_derived(P):
    """a, b, u, v (σε ps / log-λ) και ν από τις non-centered παραμέτρους."""
    P = dict(P)
    for g, z, t in (("a", "za", "tau_a"), ("b", "zb", "tau_b"), ("u", "zu", "tau_u"), ("v", "zv", "tau_v")):
        P[g] = P[t][:, None] * P[z]
    P["nu"] = 1.0 + P["nu_m1"]
    return P


class HierarchicalFit:
    """Ένα fit για μία ανάλυση: δεδομένα → NUTS → σύνοψη + ποσότητες για τα plots."""
    def __init__(self, key, label, D, grouping, n_warmup=800, n_samples=800, n_chains=2, seed=0):
        self.key, self.label, self.D, self.G = key, label, D, grouping
        self.n_warmup, self.n_samples, self.n_chains, self.seed = n_warmup, n_samples, n_chains, seed

    def run(self, cache_dir=None):
        """NUTS fit· αν υπάρχει cache_dir/<key>.npz (και ίδιος αριθμός events), φορτώνει το posterior αντί να ξανατρέξει."""
        D = self.D
        self.q, self.c = self.G(D)
        self.dt = D["tt_ps"].values.astype(np.float32)
        self.sig = D["std_ps"].values.astype(np.float32)
        cache = os.path.join(cache_dir, f"{self.key}.npz") if cache_dir else None
        if cache and os.path.exists(cache):
            z = np.load(cache)
            if int(z["n_events"]) == len(self.dt):
                self.post = {k: z[k] for k in z.files if k not in ("n_events", "diverging", "chain", "n_warmup")}
                self.diverging = z["diverging"]
                ch = z["chain"]
                self.n_chains, self.n_samples = int(ch[0]), int(ch[1])
                self.n_warmup = int(z["n_warmup"]) if "n_warmup" in z.files else self.n_warmup
                self.chains = {k: v.reshape(ch[0], ch[1], *v.shape[1:]) for k, v in self.post.items()}
                self._summarise()
                return self
        kernel = NUTS(hier_model, target_accept_prob=0.85)
        self.mcmc = MCMC(kernel, num_warmup=self.n_warmup, num_samples=self.n_samples,
                         num_chains=self.n_chains, chain_method="vectorized", progress_bar=False)
        # "vectorized" (vmap): το "parallel" (pmap σε host devices) διέρρεε μνήμη (~0.5 GB/λεπτό) σε CPU
        self.mcmc.run(jax.random.PRNGKey(self.seed), jnp.asarray(self.dt), jnp.asarray(self.sig),
                      jnp.asarray(self.q), jnp.asarray(self.c), self.G.n_q, self.G.nx * self.G.ny)
        self.post = add_derived({k: np.asarray(v) for k, v in self.mcmc.get_samples().items()})
        self.diverging = np.asarray(self.mcmc.get_extra_fields()["diverging"])
        self.chains = {k: self.post[k].reshape(self.n_chains, self.n_samples, *self.post[k].shape[1:]) for k in self.post}
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cache, n_events=len(self.dt), diverging=self.diverging, n_warmup=self.n_warmup,
                                chain=np.array([self.n_chains, self.n_samples]), **self.post)
        self._summarise()
        return self

    # ---------------- ποσότητες ----------------
    def _summarise(self):
        P, G = self.post, self.G
        self.lam_q = np.exp(P["l0"][:, None] + P["u"])               # (draws, n_q)  — μέσο κελί
        self.mu_q = P["mu0"][:, None] + P["a"]
        self.lam_c = np.exp(P["l0"][:, None] + P["v"])               # (draws, n_c)  — μέσο charge bin
        self.mu_c = P["mu0"][:, None] + P["b"]

        # no-pooling για σύγκριση: ανεξάρτητο Student-t fit (ML) σε κάθε ομάδα, με το ίδιο ν (posterior median)
        nu_fix = float(np.median(P["nu"]))
        def independent(groups, n):
            mu, lam = np.full(n, np.nan), np.full(n, np.nan)
            for j in range(n):
                m = groups == j
                if m.sum() < 20:
                    continue
                x, s = self.dt[m].astype(float), self.sig[m].astype(float)
                nll = lambda th: -np.sum(stats.t.logpdf(x, nu_fix, loc=th[0], scale=np.exp(th[1]) * s))
                r = optimize.minimize(nll, [np.median(x), 0.0], method="Nelder-Mead",
                                      options=dict(xatol=1e-4, fatol=1e-4, maxiter=2000))
                mu[j], lam[j] = r.x[0], np.exp(r.x[1])
            return mu, lam
        self.mu_q_raw, self.lam_q_raw = independent(self.q, G.n_q)
        self.mu_c_raw, self.lam_c_raw = independent(self.c, G.nx * G.ny)

        # posterior predictive coverage: ποσοστό events με |Δt − μ_i| < k·(κλίμακα), για k = 1, 2
        mu_i = np.median(P["mu0"]) + np.median(P["a"], 0)[self.q] + np.median(P["b"], 0)[self.c]
        lam_i = np.exp(np.median(P["l0"]) + np.median(P["u"], 0)[self.q] + np.median(P["v"], 0)[self.c])
        nu = float(np.median(P["nu"]))
        z_net = np.abs(self.dt) / self.sig
        z_hb = np.abs(self.dt - mu_i) / (lam_i * self.sig)
        self.coverage = {}
        for p_nom in (0.6827, 0.9545):
            k_gauss = stats.norm.ppf(0.5 + p_nom / 2)
            k_t = stats.t.ppf(0.5 + p_nom / 2, nu)
            self.coverage[f"{p_nom:.4f}"] = dict(network_gauss=float(np.mean(z_net < k_gauss)),
                                                 hierarchical_t=float(np.mean(z_hb < k_t)))
        self.z_hb, self.nu_med = z_hb, nu

        # R-hat / ESS με τα diagnostics του NumPyro (σε όλες τις παραμέτρους του sampler)
        diag = np_summary({k: self.chains[k] for k in ("mu0", "l0", "tau_a", "tau_b", "tau_u", "tau_v", "nu_m1",
                                                        "za", "zb", "zu", "zv")}, group_by_chain=True)
        rhat_max = max(float(np.nanmax(d["r_hat"])) for d in diag.values())
        ess_min = min(float(np.nanmin(d["n_eff"])) for d in diag.values())
        # ίδια diagnostics για τις ποσότητες που δείχνουμε (λ, bias ανά charge bin και ανά κελί):
        # τα μ0/l0 και οι μέσοι όροι των a,b (u,v) είναι μερικώς ανταλλάξιμα (ridge), οι συνδυασμοί όχι
        shp = (self.n_chains, self.n_samples)
        shown = {"lam_q": self.lam_q, "mu_q": self.mu_q, "lam_c": self.lam_c, "mu_c": self.mu_c}
        diag_s = np_summary({k: v.reshape(*shp, *v.shape[1:]) for k, v in shown.items()}, group_by_chain=True)
        rhat_shown = max(float(np.nanmax(d["r_hat"])) for d in diag_s.values())
        ess_shown = min(float(np.nanmin(d["n_eff"])) for d in diag_s.values())
        div = int(np.asarray(self.diverging).sum())

        def q(x):
            return dict(mean=float(np.mean(x)), sd=float(np.std(x)),
                        hdi_low=float(np.quantile(x, 0.03)), hdi_high=float(np.quantile(x, 0.97)))
        lam_cell_med = np.median(self.lam_c, 0)
        filled = G.n_cell >= 200
        self.summary = dict(
            name=self.key, label=self.label, n_events=int(len(self.dt)),
            n_charge_bins=G.n_q, n_cells=int(G.nx * G.ny), n_cells_ge200=int(filled.sum()),
            lambda0=q(np.exp(self.post["l0"])), mu0_ps=q(self.post["mu0"]),
            tau_bias_charge_ps=q(self.post["tau_a"]), tau_bias_cell_ps=q(self.post["tau_b"]),
            tau_loglambda_charge=q(self.post["tau_u"]), tau_loglambda_cell=q(self.post["tau_v"]),
            nu=q(self.post["nu"]),
            lambda_charge_range=[float(np.median(self.lam_q, 0).min()), float(np.median(self.lam_q, 0).max())],
            lambda_cell_range_filled=[float(lam_cell_med[filled].min()), float(lam_cell_med[filled].max())],
            bias_charge_range_ps=[float(np.median(self.mu_q, 0).min()), float(np.median(self.mu_q, 0).max())],
            bias_cell_range_filled_ps=[float(np.median(self.mu_c, 0)[filled].min()),
                                       float(np.median(self.mu_c, 0)[filled].max())],
            coverage=self.coverage,
            diagnostics=dict(divergences=div,
                             rhat_max=rhat_max, ess_bulk_min=ess_min,
                             rhat_max_shown=rhat_shown, ess_min_shown=ess_shown,
                             n_chains=self.n_chains, n_warmup=self.n_warmup, n_samples=self.n_samples),
        )
        return self.summary


# =====================================================================================
# Plots
# =====================================================================================
class HBPlotter:
    def __init__(self, fits, out_dir):
        self.fits, self.out_dir = fits, out_dir
        os.makedirs(out_dir, exist_ok=True)
        T = THEME                                   # ίδιο στυλ με τον Plotter του picosec_analysis.py
        plt.rcParams.update({"figure.facecolor": T["bg"], "axes.facecolor": T["bg"], "savefig.facecolor": T["bg"],
                             "text.color": T["fg"], "axes.labelcolor": T["fg"], "axes.edgecolor": T["muted"],
                             "xtick.color": T["muted"], "ytick.color": T["muted"], "grid.color": T["grid"],
                             "legend.facecolor": T["panel"], "legend.edgecolor": T["grid"], "legend.labelcolor": T["fg"],
                             "axes.titlecolor": T["fg"]})
        plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 1.0,
                             "axes.spines.top": False, "axes.spines.right": False})

    def _save(self, fig, fname):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(self.out_dir, f"{fname}.{ext}"), dpi=150, bbox_inches="tight",
                        facecolor=THEME["bg"])
        plt.close(fig)

    @staticmethod
    def _band(ax, x, draws, color, label):
        lo, med, hi = np.quantile(draws, [0.03, 0.5, 0.97], axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.30, lw=0)
        ax.plot(x, med, color=color, lw=2, label=label)

    def vs_charge(self):
        """λ και bias ανά charge bin: posterior (γραμμή + 94% band) και no-pooling (σημεία)."""
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True, gridspec_kw=dict(hspace=0.08))
        for f in self.fits:
            col = THEME["colors"][f.key]
            x = f.G.q_centre
            self._band(a1, x, f.lam_q, col, f"{f.key} · {f.label}")
            a1.plot(x, f.lam_q_raw, "o", mfc="none", mec=col, ms=5, alpha=0.8)
            self._band(a2, x, f.mu_q, col, None)
            a2.plot(x, f.mu_q_raw, "o", mfc="none", mec=col, ms=5, alpha=0.8)
        a1.axhline(1, color=THEME["zero"], ls="--", lw=1)
        a2.axhline(0, color=THEME["zero"], ls="--", lw=1)
        a1.set_ylabel("Uncertainty scale λ (core)")
        a2.set_ylabel("Bias μ [ps]"); a2.set_xlabel("Total charge Q")
        a2.set_ylim(-15, 15)
        a1.legend(loc="upper right", fontsize=9, frameon=False)
        a1.set_title("Hierarchical posterior (line, 94% band) vs independent Student-t fit per bin (○)", fontsize=11)
        self._save(fig, "hb_1_vs_charge")

    def maps(self, f):
        """Posterior λ και bias ανά κελί (hit_x, hit_y)."""
        G = f.G
        lam = np.median(f.lam_c, 0).reshape(G.ny, G.nx)
        mu = np.median(f.mu_c, 0).reshape(G.ny, G.nx)
        n = G.n_cell.reshape(G.ny, G.nx)
        lam[n == 0], mu[n == 0] = np.nan, np.nan
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(wspace=0.32))
        ext = [G.x_edges[0], G.x_edges[-1], G.y_edges[0], G.y_edges[-1]]
        im0 = axs[0].imshow(mu, origin="lower", extent=ext, cmap="RdBu_r", norm=TwoSlopeNorm(0, -10, 10), aspect="auto")
        lmin, lmax = np.nanmin(lam), np.nanmax(lam)
        span = max(abs(np.log(lmin)), abs(np.log(lmax)), 0.05)
        im1 = axs[1].imshow(lam, origin="lower", extent=ext, cmap="PuOr_r",
                            norm=TwoSlopeNorm(1.0, np.exp(-span), np.exp(span)), aspect="auto")
        for ax, im, t in ((axs[0], im0, "Bias μ [ps]"), (axs[1], im1, "Uncertainty scale λ")):
            ax.grid(False)
            ax.set_xlabel("hit_x [cm]"); ax.set_ylabel("hit_y [cm]")
            cb = fig.colorbar(im, ax=ax); cb.set_label(t)
            ax.set_title(t, fontsize=11)
        fig.suptitle(f"{f.key} · {f.label} — hierarchical posterior median per 1.3 × 1.33 mm cell", fontsize=12)
        self._save(fig, f"hb_{f.key}_2_maps")

    def shrinkage(self):
        """Independent εκτίμηση ανά κελί vs hierarchical posterior, χρώμα = πλήθος events (log)."""
        fig, axs = plt.subplots(1, len(self.fits), figsize=(5.6 * len(self.fits), 4.6), gridspec_kw=dict(wspace=0.38))
        for ax, f in zip(np.atleast_1d(axs), self.fits):
            n = f.G.n_cell
            m = n > 0
            post = np.median(f.lam_c, 0)
            sc = ax.scatter(f.lam_c_raw[m], post[m], c=np.log10(n[m]), s=14, cmap="viridis", norm=Normalize(2, 4))
            lo = np.nanmin([np.nanmin(f.lam_c_raw[m]), post[m].min()]); hi = np.nanmax([np.nanmax(f.lam_c_raw[m]), post[m].max()])
            ax.plot([lo, hi], [lo, hi], color=THEME["zero"], ls="--", lw=1)
            ax.axhline(np.median(np.exp(f.post["l0"])), color=THEME["colors"][f.key], lw=1.2)
            ax.set_xlabel("λ per cell, independent Student-t fit")
            ax.set_ylabel("λ per cell, hierarchical")
            ax.set_title(f"{f.key} · {f.label}", fontsize=10.5)
        cb = fig.colorbar(sc, ax=axs); cb.set_label("log₁₀ (events in cell)")
        fig.suptitle("Partial pooling per cell: independent fit vs hierarchical posterior (line: common λ₀)", fontsize=12)
        self._save(fig, "hb_3_shrinkage")

    def hyper(self):
        """Posterior των υπερπαραμέτρων για τις 3 αναλύσεις."""
        names = [("l0", "λ₀ = exp(l0)", np.exp), ("tau_u", "τ_u  (log λ, charge)", None), ("tau_v", "τ_v  (log λ, cell)", None),
                 ("tau_a", "τ_a  bias, charge [ps]", None), ("tau_b", "τ_b  bias, cell [ps]", None), ("nu", "ν (Student-t)", None)]
        fig, axs = plt.subplots(2, 3, figsize=(13, 6.4))
        for ax, (k, t, tr) in zip(axs.ravel(), names):
            for f in self.fits:
                x = f.post[k] if tr is None else tr(f.post[k])
                ax.hist(x, bins=40, density=True, histtype="stepfilled", alpha=0.45, color=THEME["colors"][f.key], label=f.key)
            ax.set_title(t, fontsize=11); ax.set_yticks([])
        axs[0, 0].legend(frameon=False)
        fig.suptitle("Posterior of the hyper-parameters", fontsize=12)
        fig.tight_layout()
        self._save(fig, "hb_4_hyper")

    def calibrated_pulls(self):
        """|Δt − μ_i| / (λ_i σ_i) — QQ ως προς Student-t(ν), και network pull ως προς N(0,1)."""
        fig, axs = plt.subplots(1, len(self.fits), figsize=(5.2 * len(self.fits), 4.4))
        p = np.linspace(0.01, 0.995, 200)
        for ax, f in zip(np.atleast_1d(axs), self.fits):
            z_net = np.abs(f.dt) / f.sig
            ax.plot(stats.halfnorm.ppf(p), np.quantile(z_net, p), color=THEME["muted"], lw=2, label="network σ, vs N(0,1)")
            ax.plot(np.abs(stats.t.ppf(0.5 + p / 2, f.nu_med)), np.quantile(f.z_hb, p), color=THEME["colors"][f.key], lw=2,
                    label=f"hierarchical λσ, vs t(ν={f.nu_med:.1f})")
            ax.plot([0, 6], [0, 6], color=THEME["zero"], ls="--", lw=1)
            ax.set_xlim(0, 6); ax.set_ylim(0, 6)
            ax.set_xlabel("Expected |z| quantile"); ax.set_ylabel("Observed |z| quantile")
            ax.set_aspect("equal")
            ax.set_title(f"{f.key} · {f.label}", fontsize=10.5); ax.legend(frameon=False, fontsize=9, loc="upper left")
        fig.suptitle("Calibration check (QQ of |pull|, quantiles 1%–99.5%)", fontsize=12)
        fig.tight_layout()
        self._save(fig, "hb_5_qq")

    def all(self):
        self.vs_charge()
        for f in self.fits:
            self.maps(f)
        self.shrinkage()
        self.hyper()
        self.calibrated_pulls()


# =====================================================================================
# CONFIG
# =====================================================================================
if __name__ == "__main__":
    numpyro.set_host_device_count(2)
    data_dir, out_dir = "data", "plots"
    ROOT_NOMINAL = os.path.join(data_dir, "Picosec_angelos.root")
    ROOT_NOISY = os.path.join(data_dir, "Picosec_angelos_10mV_clean.root")
    CFG = [  # (κλειδί, label, Results, ROOT, ROOT αναφοράς κανονικοποίησης, noisy reference)
        ("A1", "Train nominal / test nominal",
         os.path.join(data_dir, "Check_threshold_cut_1_q_round2__Results.parquet"), ROOT_NOMINAL, ROOT_NOMINAL, False),
        ("A2", "Train nominal / test noisy (10 mV)",
         os.path.join(data_dir, "Check_threshold_cut_1_q_round2__Results_noisy_10mV_volcorr.parquet"), ROOT_NOISY, ROOT_NOMINAL, False),
        ("A3", "Train noisy / test noisy (10 mV)",
         os.path.join(data_dir, "Noizy_train__Results.parquet"), ROOT_NOISY, ROOT_NOISY, True),
    ]
    grouping = Grouping(n_q=20)

    # προαιρετικά: python hierarchical_bayes.py A2  → κάνει μόνο το fit του A2 (σώζεται στο cache) και σταματά
    args = sys.argv[1:]
    n_draws = 800
    if "--draws" in args:                                  # π.χ. --draws 500 (λιγότερη RAM / χρόνος)
        i = args.index("--draws"); n_draws = int(args[i + 1]); del args[i:i + 2]
    only = args
    fits = []
    for key, label, res, root, ref, noisy in CFG:
        if only and key not in only:
            continue
        if not os.path.exists(res):
            res = res.replace(".parquet", ".csv")           # δέχεται και τα αρχικά csv
        a = PicosecAnalysis(key, label, res, root_path=root, norm_ref_root=ref, norm_ref_is_noisy=noisy,
                            pull_cut=np.inf).load()            # όλα τα events: τα tails τα παίρνει το Student-t
        print(f"===== {key}: {len(a.Data_test):,} events =====", flush=True)
        f = HierarchicalFit(key, label, a.Data_test, Grouping(n_q=grouping.n_q),
                             n_warmup=n_draws, n_samples=n_draws).run(cache_dir=os.path.join(data_dir, "hb_posterior"))
        print(json.dumps(f.summary, indent=2, ensure_ascii=False), flush=True)
        # αποδέσμευση μνήμης πριν την επόμενη ανάλυση (κρατάμε μόνο ό,τι χρειάζονται τα plots)
        f.D, f.mcmc, f.chains = None, None, None
        del a
        gc.collect(); jax.clear_caches()
        fits.append(f)

    if only:
        sys.exit(0)
    with open(os.path.join(out_dir, "hierarchical_summary.json"), "w") as fh:
        json.dump([f.summary for f in fits], fh, indent=2, ensure_ascii=False)
    for theme, odir in (("dark", out_dir), ("light", out_dir + "_light")):
        set_theme(theme)
        HBPlotter(fits, odir).all()
    print("Plots ->", out_dir)
