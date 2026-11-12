"""
Σχήμα της αρχιτεκτονικής του timing network, σε σκοτεινό (σελίδα) και λευκό (paper) θέμα.
Βγάζει: plots/network_architecture.{png,pdf} και plots_light/network_architecture.{png,pdf}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

from picosec_analysis import THEMES

N_HIDDEN, WIDTH, N_IN = 5, 32, 16
FEATURES = ["q", "A", "V", "t"]                    # charge, e-peak amplitude, signal max, sigmoid time
FEAT_LONG = "q: e-peak charge · A: e-peak amplitude · V: signal maximum · t: sigmoid-fit time"


def params(n_in, n_out):
    return n_in * n_out + n_out


def draw(theme_name, out_dir):
    T = THEMES[theme_name]
    acc = T["accent"] if theme_name == "dark" else "#b8860b"
    feat_col = {"q": T["colors"]["A1"], "A": T["colors"]["A2"], "V": T["colors"]["A3"], "t": acc}
    if theme_name == "dark":
        feat_col = {"q": "#f2c230", "A": "#ff6f59", "V": "#4cc9b0", "t": "#c9a7ff"}
    else:
        feat_col = {"q": "#1f5fa8", "A": "#d1495b", "V": "#2a9d5c", "t": "#7b4fb3"}
    node_face = T["panel"] if theme_name == "dark" else "#fff8e1"
    line_c = T["muted"]

    fig, ax = plt.subplots(figsize=(16.5, 7.2))
    fig.patch.set_facecolor(T["bg"]); ax.set_facecolor(T["bg"])
    ax.set_xlim(0, 16.7); ax.set_ylim(-1.75, 6.0); ax.axis("off")

    def box(x, y, w, h, ec, fc=None, lw=1.4, r=0.12):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    ec=ec, fc=fc or T["panel"], lw=lw))

    # ---------------- Inputs: 4 pads x 4 features ----------------
    in_x, chip = 0.35, 0.38
    in_nodes = []
    ax.text(1.35, 5.65, "Inputs (16)", ha="center", va="center", fontsize=13, fontweight="bold", color=T["fg"])
    for p in range(4):
        y0 = 4.55 - p * 1.25
        box(in_x, y0 - 0.12, 2.0, 0.86, ec=T["grid"] if theme_name == "dark" else "#cfcfcf")
        ax.text(in_x + 0.12, y0 + 0.58, f"pad {p}", fontsize=9.5, color=T["muted"], va="center")
        for k in range(4):        # σημεία σύνδεσης στη δεξιά πλευρά του κουτιού του pad
            in_nodes.append((in_x + 2.0, y0 + 0.02 + k * 0.2))
        for k, f in enumerate(FEATURES):
            cx = in_x + 0.28 + k * 0.48
            cy = y0 + 0.2
            ax.add_patch(FancyBboxPatch((cx - chip / 2, cy - chip / 2), chip, chip,
                                        boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=feat_col[f], ec="none", alpha=0.95))
            ax.text(cx, cy, f, ha="center", va="center", fontsize=11, fontweight="bold",
                    color="#0e0e0c" if theme_name == "dark" else "white")


    # ---------------- Hidden layers ----------------
    hx0, hdx = 3.75, 1.55
    n_draw = 8
    ys = np.linspace(4.85, 0.25, n_draw)
    layers = []
    for l in range(N_HIDDEN):
        x = hx0 + l * hdx
        box(x - 0.36, -0.12, 0.72, 5.32, ec=acc, fc=T["panel"], lw=1.2, r=0.2)
        pts = []
        for i, y in enumerate(ys):
            if i == n_draw // 2:     # ellipsis
                ax.text(x, (ys[i - 1] + y) / 2, "⋮", ha="center", va="center", fontsize=14, color=T["muted"])
                continue
            ax.add_patch(Circle((x, y), 0.17, fc=node_face, ec=acc, lw=1.3, zorder=3))
            pts.append((x, y))
        layers.append(pts)
        ax.text(x, 5.65, f"Dense {WIDTH}", ha="center", va="center", fontsize=11, fontweight="bold", color=T["fg"])
        ax.text(x, 5.33, "ReLU", ha="center", va="center", fontsize=10, color=T["muted"])
        n_in = N_IN if l == 0 else WIDTH
        ax.text(x, -0.45, f"{params(n_in, WIDTH):,} par.", ha="center", va="center", fontsize=9, color=T["muted"])

    # connections (λεπτές)
    for (x1, y1) in in_nodes:
        for (x2, y2) in layers[0]:
            ax.plot([x1 + 0.02, x2 - 0.17], [y1, y2], color=line_c, lw=0.35, alpha=0.35, zorder=1)
    for a, b in zip(layers[:-1], layers[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                ax.plot([x1 + 0.17, x2 - 0.17], [y1, y2], color=line_c, lw=0.35, alpha=0.35, zorder=1)

    # ---------------- Outputs ----------------
    ox = hx0 + N_HIDDEN * hdx + 0.25
    out_y = [3.2, 1.9]
    out_lab = ["μ", "log σ²"]
    for (x1, y1) in layers[-1]:
        for y2 in out_y:
            ax.plot([x1 + 0.17, ox - 0.33], [y1, y2], color=line_c, lw=0.45, alpha=0.45, zorder=1)
    for y, lab in zip(out_y, out_lab):
        ax.add_patch(Circle((ox, y), 0.33, fc=acc, ec=acc, lw=1.3, zorder=3))
        ax.text(ox, y, lab, ha="center", va="center", fontsize=12 if lab == "μ" else 9.5, fontweight="bold",
                color="#0e0e0c" if theme_name == "dark" else "white", zorder=4)
    ax.text(ox, 5.65, "Output", ha="center", va="center", fontsize=11, fontweight="bold", color=T["fg"])
    ax.text(ox, 5.33, "linear", ha="center", va="center", fontsize=10, color=T["muted"])
    ax.text(ox, -0.45, f"{params(WIDTH, 2):,} par.", ha="center", va="center", fontsize=9, color=T["muted"])

    # ---------------- Loss ----------------
    lx = ox + 0.85
    box(lx, 1.45, 3.9, 2.2, ec=acc, fc=T["panel"], lw=1.6)
    ax.text(lx + 1.95, 3.25, "Gaussian NLL loss", ha="center", va="center", fontsize=12, fontweight="bold", color=T["fg"])
    ax.text(lx + 1.95, 2.55, r"$\frac{1}{2}\left[\frac{(t_{true}-\mu)^2}{\sigma^2} + \log\sigma^2 + \log 2\pi\right]$",
            ha="center", va="center", fontsize=13, color=T["fg"])
    ax.text(lx + 1.95, 1.8, "σ = exp(½·log σ²): per-event uncertainty", ha="center", va="center",
            fontsize=9, color=T["muted"])
    for y in out_y:
        ax.annotate("", xy=(lx - 0.02, 2.55), xytext=(ox + 0.35, y),
                    arrowprops=dict(arrowstyle="-|>", color=acc, lw=1.4, shrinkA=0, shrinkB=0))
    ax.text(lx + 1.95, 4.25, r"target $t_{true}\sim U(-1,1)$ ns", ha="center", va="center", fontsize=10, color=T["fg"])
    ax.text(lx + 1.95, 3.95, "(added to all four pad times)", ha="center", va="center", fontsize=8.5, color=T["muted"])
    ax.annotate("", xy=(lx + 1.95, 3.67), xytext=(lx + 1.95, 3.82),
                arrowprops=dict(arrowstyle="-|>", color=T["muted"], lw=1.1))

    # ---------------- Υποσημειώσεις ----------------
    total = params(N_IN, WIDTH) + (N_HIDDEN - 1) * params(WIDTH, WIDTH) + params(WIDTH, 2)
    ax.text(0.35, -1.1, FEAT_LONG, fontsize=8.6, color=T["muted"], va="center")
    ax.text(0.35, -1.45, "q, A, V min-max normalised; t in ns. Empty pad (q < 1): q = A = 0, t = 10 ns.",
            fontsize=8.6, color=T["muted"], va="center")
    ax.text(16.55, -1.45, f"{total:,} trainable parameters · Adam, batch 512, early stopping (patience 50) · 10-fold CV",
            fontsize=8.6, color=T["muted"], va="center", ha="right")

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"network_architecture.{ext}"), dpi=170, bbox_inches="tight",
                    facecolor=T["bg"])
    plt.close(fig)
    return total


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "sans-serif"})
    n = draw("dark", "plots")
    draw("light", "plots_light")
    print("parameters:", n)
