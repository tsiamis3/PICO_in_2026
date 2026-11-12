"""
PICOSEC Micromegas — ανάλυση αποτελεσμάτων του timing network (K-Fold / ensemble).

Τρέχει τις 3 αναλύσεις (ίδια λογική με τα notebooks):
    1) nominal                      : Check_threshold_cut_1_q_round2/Results.csv
    2) train nominal / test noisy   : Check_threshold_cut_1_q_round2/Results_noisy_10mV_volcorr.csv
    3) train + test noisy           : Noizy_train/Results.csv

Αλλαγές σε σχέση με τα notebooks:
    - Όλοι οι χρόνοι (tt, std_network, bias, resolution) σε ps.
    - Πραγματικό charge: τα qall_idx_i παίρνονται από το ROOT (join με το 'srs', μοναδικό ανά event),
      με το ίδιο threshold cut του training (qall < 1 -> 0). Έλεγχος: ίδιο αποτέλεσμα με την
      αντιστροφή της min-max κανονικοποίησης.
    - Σταθερά όρια: bias στο [-10, +10] ps, κοινή κλίμακα για std/pulls σε όλες τις αναλύσεις.
    - Ενιαίο plot charge analysis (bias + resolution) για τις 3 αναλύσεις.

Τρέξιμο:
    python picosec_analysis.py            (ρυθμίσεις στο CONFIG στο τέλος)
Χρειάζεται: numpy, pandas, scipy, matplotlib, uproot, scikit-learn (για r2_score), pyarrow (αν διαβάζεις parquet)
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
from sklearn.metrics import r2_score

NS_TO_PS = 1000.0

# Σκοτεινό θέμα (μαύρο / σκούρο κίτρινο) — ίδιο με τη σελίδα
# Δύο θέματα: "dark" για τη σελίδα, "light" (λευκό) για λήψη / paper
THEMES = {
    "dark": dict(bg="#0e0e0c", panel="#16150f", fg="#e9e4d4", muted="#9d977f", grid="#2c2a22",
                 accent="#d4a017", hist="#6d6a5e", zero="#8a8470", box="#4cc9b0", cmap2d="inferno",
                 colors={"A1": "#f2c230", "A2": "#ff6f59", "A3": "#4cc9b0"}),
    "light": dict(bg="#ffffff", panel="#ffffff", fg="#1b1d21", muted="#5f6670", grid="#e4e4e4",
                  accent="#b8860b", hist="#b3b3b3", zero="#666666", box="#d1495b", cmap2d="Greys",
                  colors={"A1": "#1f5fa8", "A2": "#d1495b", "A3": "#2a9d5c"}),
}
THEME = dict(THEMES["dark"])              # το ενεργό θέμα (αλλάζει in-place με set_theme)
SCENARIO_COLORS = THEMES["dark"]["colors"]


def set_theme(name, analyses=()):
    """Αλλάζει το ενεργό θέμα και τα χρώματα των αναλύσεων (A1/A2/A3 από τα 2 πρώτα γράμματα του name)."""
    THEME.clear()
    THEME.update(THEMES[name])
    for a in analyses:
        a.color = THEME["colors"][a.name[:2]]


# =====================================================================================
# Fits — ίδια λογική με fit_gaus2 / fit_gaus2_double του pico_utils (χωρίς figures)
# =====================================================================================
def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def double_gaus(x, a1, x1, sigma1, a2, sigma2):
    return a1 * np.exp(-(x - x1) ** 2 / (2 * sigma1 ** 2)) + a2 * np.exp(-(x - x1) ** 2 / (2 * sigma2 ** 2))


class GaussFitter:
    """Gaussian fits σε histograms. Οι μονάδες είναι όποιες έχουν τα δεδομένα (εδώ ps ή αδιάστατο pull)."""

    @staticmethod
    def robust_p0(data):
        med = np.median(data)
        s = 1.4826 * np.median(np.abs(data - med))
        if not np.isfinite(s) or s <= 0:
            s = np.std(data) if np.std(data) > 0 else 1e-3
        return med, s

    @classmethod
    def fit_hist(cls, data, nbins):
        n, edges = np.histogram(data, nbins)
        bb = 0.5 * (edges[1:] + edges[:-1])
        med, s = cls.robust_p0(data)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, pcov = curve_fit(gaus, bb, n, p0=[n.max(), med, s], maxfev=5000)
            popt[2] = abs(popt[2])
            if not np.all(np.isfinite(popt)) or popt[2] == 0:
                return None
            y = gaus(bb, *popt)
            return dict(popt=popt, pcov=pcov, bb=bb, n=n, edges=edges, y=y, r2=r2_score(n, y))
        except Exception:
            return None

    @classmethod
    def best_fit(cls, data, bin_range):
        """Πολλά binnings, κρατά το καλύτερο R² (όπως _best_fit)."""
        best = None
        for nb in bin_range:
            res = cls.fit_hist(data, nb)
            if res is not None and (best is None or res["r2"] > best["r2"]):
                best = res
        if best is None:
            med, s = cls.robust_p0(data)
            n, edges = np.histogram(data, 50)
            bb = 0.5 * (edges[1:] + edges[:-1])
            popt = np.array([n.max(), med, s])
            best = dict(popt=popt, pcov=np.full((3, 3), np.nan), bb=bb, n=n, edges=edges,
                        y=gaus(bb, *popt), r2=np.nan)
        return best

    @classmethod
    def single(cls, data):
        """fit_gaus2: 1ο fit σε όλα, 2ο fit στο ±5σ. Επιστρέφει mu, sigma, err_mu, err_sigma, data στο ±5σ."""
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        f1 = cls.best_fit(data, range(20, 101, 5))
        mm, ss = f1["popt"][1], f1["popt"][2]
        data2 = data[(data > mm - 5 * ss) & (data < mm + 5 * ss)]
        if len(data2) < 20:
            data2 = data
        f2 = cls.best_fit(data2, range(10, 101, 5))
        perr = np.sqrt(np.abs(np.diag(f2["pcov"])))
        return dict(mu=f2["popt"][1], sigma=f2["popt"][2], err_mu=perr[1], err_sigma=perr[2],
                    data=data2, fit=f2)

    @classmethod
    def double(cls, data, nbins, min_frac=0.10):
        """
        fit_gaus2_double: single fit -> δεδομένα στο ±5σ -> double Gaussian (κοινό mean).
        σ_tot = sqrt(f1 σ1² + f2 σ2²), με f_i τα κλάσματα ΕΜΒΑΔΟΥ (a_i σ_i).
        """
        s1 = cls.single(data)
        d = s1["data"]
        n, edges = np.histogram(d, nbins)
        bb = 0.5 * (edges[1:] + edges[:-1])
        med, s = cls.robust_p0(d)
        p0 = [0.8 * n.max(), med, s, 0.2 * n.max(), 2 * s]
        lo = (0, d.min(), 0, 0, 0)
        hi = (np.inf, d.max(), 10 * np.std(d), np.inf, 10 * np.std(d))
        popt, pcov = curve_fit(double_gaus, bb, n, p0=p0, bounds=(lo, hi), maxfev=20000)
        a1, mu, sg1, a2, sg2 = popt
        # το "core" να είναι πάντα η στενή Gaussian
        if sg2 < sg1:
            a1, sg1, a2, sg2 = a2, sg2, a1, sg1
        w1, w2 = a1 * sg1, a2 * sg2
        f1, f2 = w1 / (w1 + w2), w2 / (w1 + w2)
        sig_tot = np.sqrt(f1 * sg1 ** 2 + f2 * sg2 ** 2)
        perr = np.sqrt(np.abs(np.diag(pcov)))
        if min(f1, f2) < min_frac:
            # Η δεύτερη Gaussian δεν χρειάζεται (εκφυλισμένο fit, συνήθως σε μικρή στατιστική):
            # κρατάμε τη single Gaussian στο ±5σ.
            f = s1["fit"]
            a0, mu, sg = f["popt"]
            a0 = a0 * (edges[1] - edges[0]) / (f["edges"][1] - f["edges"][0])   # ίδιο binning με το plot
            return dict(mu=mu, err_mu=s1["err_mu"], sigma_core=sg, sigma_tail=np.nan, frac_core=1.0,
                        sigma_tot=sg, sigma_single=sg, mu_single=mu, rms=np.std(d), n_events=len(d),
                        hist=dict(n=n, edges=edges), popt=(a0, mu, sg, 0.0, sg), single=True)
        return dict(mu=mu, err_mu=perr[1], sigma_core=sg1, sigma_tail=sg2, frac_core=f1, single=False,
                    sigma_tot=sig_tot, sigma_single=s1["sigma"], mu_single=s1["mu"],
                    rms=np.std(d), n_events=len(d),
                    hist=dict(n=n, edges=edges), popt=(a1, mu, sg1, a2, sg2))


# =====================================================================================
# Επιλογή: tracks στα κέντρα των pads
# =====================================================================================
class PadCenterSelection:
    """
    Κρατά τα tracks με |hit_x - x_c| <= half_width ΚΑΙ |hit_y - y_c| <= half_width για κάποιο κέντρο pad
    (τετράγωνο παράθυρο). hit_x, hit_y σε cm.
    """
    def __init__(self, x_centers, y_centers, half_width_cm=0.125):
        self.xc, self.yc = np.asarray(x_centers, float), np.asarray(y_centers, float)
        self.hw = half_width_cm
        # heatmap: ένα κελί ανά κέντρο pad (3 στήλες x 2 γραμμές)
        xs, ys = np.unique(self.xc), np.unique(self.yc)
        self.x_edges = np.concatenate([[xs[0] - 0.5], 0.5 * (xs[1:] + xs[:-1]), [xs[-1] + 0.5]])
        self.y_edges = np.concatenate([[ys[0] - 0.5], 0.5 * (ys[1:] + ys[:-1]), [ys[-1] + 0.5]])

    def __call__(self, D):
        m = np.zeros(len(D), dtype=bool)
        for x, y in zip(self.xc, self.yc):
            m |= (np.abs(D["hit_x"] - x) <= self.hw).values & (np.abs(D["hit_y"] - y) <= self.hw).values
        return m

    def describe(self):
        return f"|Δx|, |Δy| ≤ {10*self.hw:.2f} mm from a pad centre"


# =====================================================================================
# Μία ανάλυση
# =====================================================================================
class PicosecAnalysis:
    """
    Φορτώνει ένα Results αρχείο, ανακτά το πραγματικό charge από το ROOT, μετατρέπει σε ps και
    υπολογίζει: total resolution, pulls, charge analysis, heatmaps (tt και pull).
    """
    pads = range(4)

    def __init__(self, name, label, results_path, root_path, norm_ref_root, norm_ref_is_noisy,
                 thres=1.0, pull_cut=5.0, events_per_charge_bin=10000,
                 x_edges=None, y_edges=None, min_hits=200, color="C0", marker="o",
                 selection=None, selection_label="All tracks"):
        self.name, self.label = name, label
        self.selection, self.selection_label = selection, selection_label
        if selection is not None and x_edges is None:
            x_edges, y_edges = selection.x_edges, selection.y_edges
        self.results_path, self.root_path = results_path, root_path
        self.norm_ref_root, self.norm_ref_is_noisy = norm_ref_root, norm_ref_is_noisy
        self.thres, self.pull_cut = thres, pull_cut
        self.events_per_charge_bin = events_per_charge_bin
        self.x_edges = np.linspace(-1.3, 1.3, 11) if x_edges is None else x_edges
        self.y_edges = np.linspace(-0.8, 0.8, 6) if y_edges is None else y_edges
        self.min_hits = min_hits
        self.color, self.marker = color, marker
        self.summary = {"name": name, "label": label, "selection": selection_label,
                        "events_per_charge_point": events_per_charge_bin}

    @property
    def title(self):
        return self.label if self.selection is None else f"{self.label} · pad centres"

    # ---------------- Φόρτωση ----------------
    @staticmethod
    def _read_table(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _root_after_threshold(self, path):
        """ROOT -> DataFrame με το ίδιο threshold cut του training (qall < thres -> 0)."""
        FF = uproot.open(path)["events;1"].arrays(library="pd")
        for i in self.pads:
            FF.loc[FF[f"qall_idx_{i}"] < self.thres, f"qall_idx_{i}"] = 0.0
        return FF

    def load(self):
        Data_test = self._read_table(self.results_path)
        Data_test = Data_test.loc[:, ~Data_test.columns.str.startswith("Unnamed")]
        self.summary["n_events_results"] = int(len(Data_test))

        # ---- Πραγματικό charge: join με το ROOT στο 'srs' ----
        raw = self._root_after_threshold(self.root_path)
        assert raw["srs"].is_unique, "Το srs δεν είναι μοναδικό στο ROOT"
        qcols = [f"qall_idx_{i}" for i in self.pads]
        raw = raw[["srs"] + qcols].rename(columns={c: c + "_raw" for c in qcols})
        Data_test["srs"] = Data_test["srs"].round().astype(np.int64)
        raw["srs"] = raw["srs"].astype(np.int64)
        Data_test = Data_test.merge(raw, on="srs", how="left", validate="one_to_one")
        n_missing = int(Data_test["qall_idx_0_raw"].isna().sum())
        assert n_missing == 0, f"{n_missing} events δεν βρέθηκαν στο ROOT"
        Data_test["Q"] = Data_test[[c + "_raw" for c in qcols]].sum(axis=1)

        # ---- Έλεγχος: αντιστροφή της min-max (norm_scenario=2) ----
        ref = self._root_after_threshold(self.norm_ref_root)
        if self.norm_ref_is_noisy:          # ίδια βήματα με το Noizy_train.py (rename + dropna)
            rename_map = {}
            for i in self.pads:
                rename_map[f"vol_max_waveform_{i}"] = f"vol_max_{i}"
                rename_map[f"sigmoid_timepoint_{i}"] = f"time_fit_{i}"
            ref = ref.drop(columns=list(rename_map.values())).rename(columns=rename_map)
            inp = [f"{p}_{i}" for p in ("qall_idx", "epeak", "vol_max", "time_fit") for i in self.pads]
            ref = ref.dropna(subset=inp)
        q_inv = np.zeros(len(Data_test))
        for c in qcols:
            mn, mx = ref[c].min(), ref[c].max()
            q_inv += Data_test[c].values * (mx - mn) + mn
        dq = np.abs(q_inv - Data_test["Q"].values)
        self.summary["charge_check_max_abs_diff"] = float(np.nanmax(dq))
        self.summary["charge_check_median_abs_diff"] = float(np.nanmedian(dq))

        # ---- ns -> ps και pull ----
        Data_test["tt_ps"] = Data_test["tt"] * NS_TO_PS
        Data_test["std_ps"] = Data_test["std_network"] * NS_TO_PS
        Data_test["pull"] = Data_test["tt"] / Data_test["std_network"]
        if self.selection is not None:
            m = self.selection(Data_test)
            self.summary["frac_selected"] = float(m.mean())
            Data_test = Data_test[m].copy()
        n0 = len(Data_test)
        Data_test = Data_test[Data_test["pull"].abs() < self.pull_cut].copy()
        self.summary["pull_cut"] = self.pull_cut
        self.summary["frac_removed_by_pull_cut"] = float(1 - len(Data_test) / n0)
        self.summary["n_events"] = int(len(Data_test))
        self.Data_test = Data_test
        return self

    # ---------------- Υπολογισμοί ----------------
    def total_resolution(self, nbins=500):
        self.res_tt = GaussFitter.double(self.Data_test["tt_ps"].values, nbins)
        self.summary["resolution_ps"] = {k: float(v) for k, v in self.res_tt.items()
                                         if k in ("mu", "err_mu", "sigma_core", "sigma_tail", "frac_core",
                                                  "sigma_tot", "sigma_single", "mu_single", "rms")}
        return self.res_tt

    def pulls(self, nbins=300):
        self.res_pull = GaussFitter.double(self.Data_test["pull"].values, nbins)
        self.summary["pull"] = {k: float(v) for k, v in self.res_pull.items()
                                if k in ("mu", "err_mu", "sigma_core", "sigma_tail", "frac_core",
                                         "sigma_tot", "sigma_single", "mu_single", "rms")}
        return self.res_pull

    def charge_analysis(self):
        """Ίδια λογική με Charge_Analysis: ταξινόμηση ως προς Q, κομμάτια των N events, Gaussian fit ανά κομμάτι."""
        D = self.Data_test.sort_values("Q").reset_index(drop=True)
        N = self.events_per_charge_bin
        k = len(D) // N
        chunks = [D.iloc[i * N:(i + 1) * N] for i in range(k)]
        if len(D) - k * N > 100:            # τα τελευταία events
            chunks.append(D.iloc[k * N:])
        rows = []
        for ch in chunks:
            f = GaussFitter.single(ch["tt_ps"].values)
            fp = GaussFitter.single(ch["pull"].values)
            rows.append(dict(Q=ch["Q"].mean(), Q_min=ch["Q"].min(), Q_max=ch["Q"].max(), N=len(ch),
                             Mu=f["mu"], Mu_error=f["err_mu"], Sigma=f["sigma"], Sigma_error=f["err_sigma"],
                             Pull_mu=fp["mu"], Pull_sigma=fp["sigma"]))
        self.charge_frame = pd.DataFrame(rows)
        return self.charge_frame

    def _map(self, values):
        D = self.Data_test
        nx, ny = len(self.x_edges) - 1, len(self.y_edges) - 1
        ix = np.clip(np.digitize(D["hit_x"], self.x_edges) - 1, 0, nx - 1)
        iy = np.clip(np.digitize(D["hit_y"], self.y_edges) - 1, 0, ny - 1)
        mu_map, sd_map = np.full((ny, nx), np.nan), np.full((ny, nx), np.nan)
        n_map = np.zeros((ny, nx), dtype=int)
        v = np.asarray(values)
        for i in range(nx):
            for j in range(ny):
                sel = v[(ix == i) & (iy == j)]
                n_map[j, i] = len(sel)
                if len(sel) < self.min_hits:
                    continue
                f = GaussFitter.single(sel)
                mu_map[j, i], sd_map[j, i] = f["mu"], f["sigma"]
        return dict(mu=mu_map, sd=sd_map, n=n_map)

    def heatmaps(self):
        self.map_tt = self._map(self.Data_test["tt_ps"].values)
        self.map_pull = self._map(self.Data_test["pull"].values)
        return self.map_tt, self.map_pull

    def run(self):
        self.load()
        self.total_resolution()
        self.pulls()
        self.charge_analysis()
        self.heatmaps()
        return self


# =====================================================================================
# Plots
# =====================================================================================
class Plotter:
    """Όλα τα plots, με κοινά όρια μεταξύ αναλύσεων."""

    def __init__(self, analyses, out_dir, bias_lim_ps=10.0, q_unit="",
                 pull_bias_lim=0.5, pull_sd_lim=(0.0, 2.0), sd_lim=None, suffix="", comparison_title=None):
        self.analyses, self.out_dir = analyses, out_dir
        self.suffix = suffix
        self.comparison_title = comparison_title or "Bias and resolution vs charge — comparison"
        self.bias_lim = bias_lim_ps
        self.q_label = "Total charge Q" + (f" [{q_unit}]" if q_unit else "")
        self.pull_bias_lim, self.pull_sd_lim = pull_bias_lim, pull_sd_lim
        os.makedirs(out_dir, exist_ok=True)
        T = THEME
        plt.rcParams.update({"figure.facecolor": T["bg"], "axes.facecolor": T["bg"], "savefig.facecolor": T["bg"],
                             "text.color": T["fg"], "axes.labelcolor": T["fg"], "axes.edgecolor": T["muted"],
                             "xtick.color": T["muted"], "ytick.color": T["muted"], "grid.color": T["grid"],
                             "legend.facecolor": T["panel"], "legend.edgecolor": T["grid"], "legend.labelcolor": T["fg"],
                             "axes.titlecolor": T["fg"]})
        plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 1.0,
                             "axes.spines.top": False, "axes.spines.right": False})
        # κοινή κλίμακα για τα std heatmaps (ps)
        if sd_lim is None:
            sds = np.concatenate([a.map_tt["sd"].ravel() for a in analyses])
            sd_lim = (np.floor(np.nanmin(sds)), np.ceil(np.nanmax(sds)))
        self.sd_lim = sd_lim

    def _save(self, fig, fname):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(self.out_dir, f"{fname}.{ext}"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- 1, 2) Total resolution και pulls ----
    def _double_fit_plot(self, a, res, xlabel, unit, fname, title):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        n, edges = res["hist"]["n"], res["hist"]["edges"]
        bb = 0.5 * (edges[1:] + edges[:-1])
        ax.stairs(n, edges, color=THEME["hist"], fill=True, alpha=0.8, label="Data")
        a1, mu, s1, a2, s2 = res["popt"]
        xf = np.linspace(edges[0], edges[-1], 1000)
        fmt = "{:.2f}" if unit == "ps" else "{:.3f}"
        u = f" {unit}" if unit else ""
        if res.get("single"):
            ax.plot(xf, gaus(xf, a1, mu, s1), color=a.color, lw=2, label="Gaussian")
            txt = (f"μ = {fmt.format(res['mu'])}{u}\nσ = {fmt.format(res['sigma_tot'])}{u}\n"
                   f"N = {res['n_events']:,}\n(single Gaussian:\n 2nd component < 10%)")
        else:
            ax.plot(xf, double_gaus(xf, *res["popt"]), color=a.color, lw=2, label="Double Gaussian")
            ax.plot(xf, gaus(xf, a1, mu, s1), "--", color=a.color, lw=1.2, alpha=0.8, label="core")
            ax.plot(xf, gaus(xf, a2, mu, s2), ":", color=a.color, lw=1.2, alpha=0.8, label="tail")
            txt = (f"μ = {fmt.format(res['mu'])}{u}\n"
                   f"σ_core = {fmt.format(res['sigma_core'])}{u} ({100*res['frac_core']:.0f}%)\n"
                   f"σ_tail = {fmt.format(res['sigma_tail'])}{u}\n"
                   f"σ_tot = {fmt.format(res['sigma_tot'])}{u}\n"
                   f"N = {res['n_events']:,}")
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=11,
                bbox=dict(boxstyle="round", fc=THEME["panel"], ec=THEME["grid"]))
        ax.set_xlabel(xlabel); ax.set_ylabel("Events / bin")
        ax.set_title(f"{title} — {a.title}")
        ax.legend(loc="upper right", fontsize=10)
        lim = max(abs(mu - 5 * res["sigma_tot"]), abs(mu + 5 * res["sigma_tot"]))
        lim = min(lim, max(abs(edges[0]), abs(edges[-1])))     # όχι πέρα από τα δεδομένα
        ax.set_xlim(-lim, lim)
        self._save(fig, fname)

    def total_resolution(self, a):
        self._double_fit_plot(a, a.res_tt, "t_network − t_true [ps]", "ps",
                              f"{a.name}_1_total_resolution", "Total resolution")

    def pulls(self, a):
        self._double_fit_plot(a, a.res_pull, "Pull = (t_network − t_true) / σ_network", "",
                              f"{a.name}_2_pulls", "Pull")

    # ---- 3) Charge analysis ανά ανάλυση ----
    def _charge_axes(self, ax_mu, ax_sd, a, fr):
        ax_mu.errorbar(fr["Q"], fr["Mu"], yerr=fr["Mu_error"], fmt=a.marker, ms=5, color=a.color,
                       mfc=THEME["bg"] if a.marker != "o" else a.color, label=a.label, capsize=2)
        ax_sd.errorbar(fr["Q"], fr["Sigma"], yerr=fr["Sigma_error"], fmt=a.marker, ms=5, color=a.color,
                       mfc=THEME["bg"] if a.marker != "o" else a.color, label=a.label, capsize=2)

    def _charge_figure(self, analyses, fname, title):
        fig, (ax_mu, ax_sd) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                           gridspec_kw=dict(height_ratios=[1, 1.4], hspace=0.06))
        for a in analyses:
            self._charge_axes(ax_mu, ax_sd, a, a.charge_frame)
        ax_mu.axhline(0, color=THEME["zero"], lw=1)
        ax_mu.set_ylim(-self.bias_lim, self.bias_lim)
        ax_mu.set_ylabel("Bias μ [ps]")
        ax_sd.set_ylabel("Resolution σ [ps]")
        ax_sd.set_xlabel(self.q_label)
        ax_sd.set_ylim(bottom=0)
        ax_mu.set_title(title)
        if len(analyses) > 1:
            ax_sd.legend(loc="upper right")
        else:
            fr = analyses[0].charge_frame
            ax_sd.text(0.98, 0.95, f"min σ = {fr['Sigma'].min():.2f} ps\n{analyses[0].events_per_charge_bin:,} events/point",
                       transform=ax_sd.transAxes, ha="right", va="top",
                       bbox=dict(boxstyle="round", fc=THEME["panel"], ec=THEME["grid"]))
        self._save(fig, fname)

    def charge_analysis(self, a):
        self._charge_figure([a], f"{a.name}_3_charge_analysis", f"Charge analysis — {a.title}")

    def charge_comparison(self):
        self._charge_figure(self.analyses, f"comparison_charge_analysis{self.suffix}", self.comparison_title)

    # ---- 4, 5) Heatmaps ----
    def _heatmap_pair(self, a, m, fname, what, unit, mu_lim, sd_norm, sd_cmap, fmt):
        x_e, y_e = a.x_edges, a.y_edges
        xc, yc = 0.5 * (x_e[:-1] + x_e[1:]), 0.5 * (y_e[:-1] + y_e[1:])
        fig, axs = plt.subplots(2, 1, figsize=(13, 10.5))
        u = f" [{unit}]" if unit else ""
        panels = [(m["mu"], "coolwarm", plt.Normalize(-mu_lim, mu_lim), f"Bias of {what}{u}"),
                  (m["sd"], sd_cmap, sd_norm, f"Std of {what}{u}")]
        for ax, (data, cmap_name, norm, title) in zip(axs, panels):
            cmap = plt.get_cmap(cmap_name)
            pm = ax.pcolormesh(x_e, y_e, data, cmap=cmap, norm=norm)
            cb = fig.colorbar(pm, ax=ax, extend="both")
            cb.set_label(title)
            for j in range(len(yc)):
                for i in range(len(xc)):
                    v = data[j, i]
                    if np.isnan(v):
                        continue
                    r, g, b, _ = cmap(norm(v))
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    label = fmt.format(v) + (f"\nN = {m['n'][j, i]:,}" if a.selection is not None else "")
                    ax.text(xc[i], yc[j], label, ha="center", va="center", fontsize=10 if a.selection is None else 13,
                            fontweight="bold", color="black" if lum > 0.5 else "white")
            ax.set_title(f"{title} — {a.title}")
            ax.set_xlabel("hit_x"); ax.set_ylabel("hit_y")
            ax.grid(False)
            if a.selection is None:
                ax.set_xticks(x_e); ax.set_yticks(y_e)
                ax.tick_params(labelsize=9)
                plt.setp(ax.get_xticklabels(), rotation=45)
            else:   # ένα κελί ανά κέντρο pad: ticks στα κέντρα
                ax.set_xticks(xc); ax.set_yticks(yc)
                ax.set_xlabel("pad centre hit_x [cm]"); ax.set_ylabel("pad centre hit_y [cm]")
        fig.tight_layout()
        self._save(fig, fname)

    def heatmaps_tt(self, a):
        self._heatmap_pair(a, a.map_tt, f"{a.name}_4_heatmap_tt", "t_network − t_true", "ps",
                           self.bias_lim, plt.Normalize(*self.sd_lim), "viridis", "{:.1f}")

    def heatmaps_pull(self, a):
        lo, hi = self.pull_sd_lim
        self._heatmap_pair(a, a.map_pull, f"{a.name}_5_heatmap_pull", "pull", "",
                           self.pull_bias_lim, plt.Normalize(lo, hi), "PiYG_r", "{:.2f}")

    def selection_map(self, a, sel, fname="selection_pad_centres"):
        """hit_x/hit_y όλων των tracks, με τα παράθυρα της επιλογής."""
        D = a.Data_test
        fig, ax = plt.subplots(figsize=(11, 6.5))
        h = ax.hist2d(D["hit_x"], D["hit_y"], bins=[260, 160], cmap=THEME["cmap2d"], cmin=1)
        fig.colorbar(h[3], ax=ax, label="Tracks / bin")
        for x, y in zip(sel.xc, sel.yc):
            ax.add_patch(plt.Rectangle((x - sel.hw, y - sel.hw), 2 * sel.hw, 2 * sel.hw,
                                       fill=False, ec=THEME["box"], lw=2))
            ax.plot(x, y, "+", color=THEME["box"], ms=10, mew=2)
        ax.set_xlabel("hit_x [cm]"); ax.set_ylabel("hit_y [cm]")
        ax.set_aspect("equal"); ax.grid(False)
        ax.set_title(f"Track positions ({a.label})\nboxes: {sel.describe()}", fontsize=12)
        self._save(fig, fname)

    def all(self):
        for a in self.analyses:
            self.total_resolution(a)
            self.pulls(a)
            self.charge_analysis(a)
            self.heatmaps_tt(a)
            self.heatmaps_pull(a)
        self.charge_comparison()


# =====================================================================================
# CONFIG
# =====================================================================================
if __name__ == "__main__":
    data_dir = "data"            # φάκελος με ROOT + Results
    out_dir = "plots"

    ROOT_NOMINAL = os.path.join(data_dir, "Picosec_angelos.root")
    ROOT_NOISY = os.path.join(data_dir, "Picosec_angelos_10mV_clean.root")

    analyses = [
        PicosecAnalysis("A1_nominal", "Train nominal / test nominal",
                        os.path.join(data_dir, "Check_threshold_cut_1_q_round2__Results.parquet"),
                        root_path=ROOT_NOMINAL, norm_ref_root=ROOT_NOMINAL, norm_ref_is_noisy=False,
                        color=SCENARIO_COLORS["A1"], marker="o"),
        PicosecAnalysis("A2_train_nominal_test_noisy", "Train nominal / test noisy (10 mV)",
                        os.path.join(data_dir, "Check_threshold_cut_1_q_round2__Results_noisy_10mV_volcorr.parquet"),
                        root_path=ROOT_NOISY, norm_ref_root=ROOT_NOMINAL, norm_ref_is_noisy=False,
                        color=SCENARIO_COLORS["A2"], marker="s"),
        PicosecAnalysis("A3_noisy_train", "Train noisy / test noisy (10 mV)",
                        os.path.join(data_dir, "Noizy_train__Results.parquet"),
                        root_path=ROOT_NOISY, norm_ref_root=ROOT_NOISY, norm_ref_is_noisy=True,
                        color=SCENARIO_COLORS["A3"], marker="^"),
    ]
    analyses = [a for a in analyses if os.path.exists(a.results_path)]

    # ---- Ίδιες αναλύσεις, μόνο για tracks στα κέντρα των pads ----
    pad_centres = PadCenterSelection(x_centers=[1.002, 1.002, 0, 0, -1.002, -1.002],
                                     y_centers=[0.501, -0.501, 0.501, -0.501, 0.501, -0.501],
                                     half_width_cm=0.125)          # ±1.25 mm
    analyses_c = [PicosecAnalysis(a.name + "_centres", a.label, a.results_path, a.root_path, a.norm_ref_root,
                                  a.norm_ref_is_noisy, events_per_charge_bin=1000, color=a.color, marker=a.marker,
                                  selection=pad_centres, selection_label="Pad centres (±1.25 mm)")
                  for a in analyses]

    for a in analyses + analyses_c:
        print(f"\n===== {a.title} =====")
        a.run()
        print(json.dumps(a.summary, indent=2, ensure_ascii=False))

    # κοινή κλίμακα std για όλα τα heatmaps (όλα τα tracks + κέντρα)
    sds = np.concatenate([a.map_tt["sd"].ravel() for a in analyses + analyses_c])
    sd_lim = (np.floor(np.nanmin(sds)), np.ceil(np.nanmax(sds)))

    # ίδια plots σε δύο θέματα: σκοτεινό (σελίδα) και λευκό (λήψη / paper)
    for theme, odir in (("dark", out_dir), ("light", out_dir + "_light")):
        set_theme(theme, analyses + analyses_c)
        P = Plotter(analyses, odir, bias_lim_ps=10.0, q_unit="", sd_lim=sd_lim)
        P.all()
        P.selection_map(analyses[0], pad_centres)
        Pc = Plotter(analyses_c, odir, bias_lim_ps=10.0, q_unit="", sd_lim=sd_lim, suffix="_centres",
                     comparison_title="Bias and resolution vs charge — pad centres (±1.25 mm)")
        Pc.all()
    set_theme("dark", analyses + analyses_c)

    for a in analyses + analyses_c:
        a.charge_frame.to_csv(os.path.join(out_dir, f"{a.name}_charge_analysis.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump([a.summary for a in analyses + analyses_c], f, indent=2, ensure_ascii=False)
    print("\nPlots ->", out_dir)
