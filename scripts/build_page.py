"""
Φτιάχνει ένα αυτόνομο index.html (εικόνες ενσωματωμένες ως base64) με όλα τα αποτελέσματα.
Τρέξιμο:  python build_page.py   (μετά το picosec_analysis.py)
Για GitHub Pages: ανέβασε το index.html στο repo (root ή /docs) και ενεργοποίησε το Pages.
"""
import os
import json
import base64
import html
import pandas as pd

PLOTS = "plots"
OUT = "site/index.html"

analyses = json.load(open(os.path.join(PLOTS, "summary.json")))
STYLE = {  # ίδια χρώματα με τα PNG
    "A1_nominal": ("#1f5fa8", "circle"),
    "A2_train_nominal_test_noisy": ("#d1495b", "square-open"),
    "A3_noisy_train": ("#2a9d5c", "triangle-up-open"),
}
for k in list(STYLE):
    STYLE[k + "_centres"] = STYLE[k]
SHORT = {"A1_nominal": "A1", "A2_train_nominal_test_noisy": "A2", "A3_noisy_train": "A3"}
for k in list(SHORT):
    SHORT[k + "_centres"] = SHORT[k]
SEL = lambda a: "c" if a["name"].endswith("_centres") else "all"
SEL_TITLE = {"all": "All tracks", "c": "Pad centres (±1.25 mm)"}
PLOT_KINDS = [
    ("1_total_resolution", "Total resolution", "Double-Gaussian fit of t<sub>network</sub> − t<sub>true</sub>."),
    ("2_pulls", "Pulls", "(t<sub>network</sub> − t<sub>true</sub>) / σ<sub>network</sub>; a well-calibrated network gives σ ≈ 1."),
    ("3_charge_analysis", "Charge analysis", "Events sorted by total charge Q, {epp:,} events per point, Gaussian fit per point."),
    ("4_heatmap_tt", "Heatmaps: bias and std", "Gaussian fit of t<sub>network</sub> − t<sub>true</sub> per {cell} (≥ 200 hits)."),
    ("5_heatmap_pull", "Heatmaps: pull bias and std", "Gaussian fit of the pull per {cell} (≥ 200 hits)."),
]


def img(fname, alt):
    with open(os.path.join(PLOTS, fname), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img loading="lazy" src="data:image/png;base64,{b64}" alt="{html.escape(alt)}">'


# ---------------- Πίνακας σύνοψης ----------------
rows = []
for sel in ("all", "c"):
    rows.append(f'<tr class="group"><td colspan="11">{SEL_TITLE[sel]}</td></tr>')
    for a in [a for a in analyses if SEL(a) == sel]:
        r, p = a["resolution_ps"], a["pull"]
        c = STYLE[a["name"]][0]
        rows.append(f"""<tr>
      <td><span class="dot" style="background:{c}"></span><b>{SHORT[a['name']]}</b> {html.escape(a['label'])}</td>
      <td class="num">{a['n_events']:,}</td>
      <td class="num">{r['sigma_core']:.1f}</td>
      <td class="num">{'–' if r['sigma_tail'] != r['sigma_tail'] else f"{r['sigma_tail']:.1f}"}</td>
      <td class="num">{100*r['frac_core']:.0f}%</td>
      <td class="num"><b>{r['sigma_tot']:.1f}</b></td>
      <td class="num">{r['sigma_single']:.1f}</td>
      <td class="num">{r['mu']:+.1f}</td>
      <td class="num">{p['sigma_tot']:.2f}</td>
      <td class="num">{p['mu']:+.3f}</td>
      <td class="num">{100*a['frac_removed_by_pull_cut']:.2f}%</td>
    </tr>""")
table = "\n".join(rows)

# ---------------- Δεδομένα για το διαδραστικό charge plot ----------------
traces = []
for a in analyses:
    fr = pd.read_csv(os.path.join(PLOTS, f"{a['name']}_charge_analysis.csv"))
    c, sym = STYLE[a["name"]]
    for yaxis, y, err in (("y", "Mu", "Mu_error"), ("y2", "Sigma", "Sigma_error")):
        traces.append(dict(
            x=fr["Q"].round(3).tolist(), y=fr[y].round(3).tolist(),
            error_y=dict(type="data", array=fr[err].round(3).tolist(), thickness=1.2, width=3, color=c),
            customdata=fr[["Q_min", "Q_max", "N"]].round(2).values.tolist(),
            mode="markers", marker=dict(color=c, symbol=sym, size=7, line=dict(color=c, width=1.5)),
            name=a["label"], legendgroup=a["name"], showlegend=(yaxis == "y2"), yaxis=yaxis,
            meta=SEL(a), visible=(SEL(a) == "all"),
            hovertemplate=("<b>%{fullData.name}</b><br>Q = %{x:.1f} (range %{customdata[0]:.1f}–%{customdata[1]:.1f})"
                           "<br>" + ("bias" if y == "Mu" else "σ") + " = %{y:.2f} ± %{error_y.array:.2f} ps"
                           "<br>N = %{customdata[2]:,}<extra></extra>"),
        ))

# ---------------- Ενότητες ανά ανάλυση ----------------
sections = []
for a in analyses:
    sel, sid = SEL(a), SHORT[a["name"]].lower() + ("c" if SEL(a) == "c" else "")
    cell = "pad centre" if sel == "c" else "(hit_x, hit_y) cell"
    figs = []
    if sel == "c":
        figs.append(f"""<figure class="wide">
          <figcaption><h3>Selection</h3><p>Track positions (nominal data) and the six ±1.25&nbsp;mm windows around the pad centres.</p></figcaption>
          {img("selection_pad_centres.png", "Pad centre selection")}
        </figure>""")
    for kind, title, cap in PLOT_KINDS:
        cap = cap.format(epp=a["events_per_charge_point"], cell=cell)
        figs.append(f"""<figure class="{'wide' if 'heatmap' in kind and sel == 'all' else ''}">
          <figcaption><h3>{title}</h3><p>{cap}</p></figcaption>
          {img(f"{a['name']}_{kind}.png", f"{title} — {a['label']}")}
        </figure>""")
    r = a["resolution_ps"]
    extra = f"<div><span>selected</span><b>{100*a['frac_selected']:.1f}%</b></div>" if sel == "c" else ""
    sections.append(f"""<section class="analysis" id="{sid}">
      <div class="kpis">
        <div><span>σ<sub>tot</sub></span><b>{r['sigma_tot']:.1f} ps</b></div>
        <div><span>σ<sub>core</sub></span><b>{r['sigma_core']:.1f} ps</b></div>
        <div><span>bias</span><b>{r['mu']:+.1f} ps</b></div>
        <div><span>σ(pull)</span><b>{a['pull']['sigma_tot']:.2f}</b></div>
        <div><span>events</span><b>{a['n_events']:,}</b></div>{extra}
      </div>
      <div class="grid">{''.join(figs)}</div>
    </section>""")


NET_SECTION = """<section class="analysis" id="net">
  <div style="text-align:right;margin-bottom:8px"><button id="copyNet">Copy text</button></div>
  <div class="card prose" id="netText">
    <h3>Network</h3>
    <p>The event time is reconstructed by a fully connected neural network with a heteroscedastic Gaussian output.
       The input is a 16-dimensional vector: for each of the four readout pads, the e-peak charge (<code>qall</code>),
       the e-peak amplitude (<code>epeak</code>), the signal maximum (<code>vol_max</code>) and the signal arrival time from a
       sigmoid fit (<code>time_fit</code>). Pads with charge below 1 are treated as empty: their charge and amplitude are set to 0
       and their time to a fixed out-of-range value. Charges, amplitudes and voltages are min-max normalised per variable;
       times are used in ns without normalisation.</p>
    <p>The network has five hidden layers of 32 ReLU units (≈4.8k parameters) and two outputs, the predicted time μ and
       log σ², so that every event gets its own time estimate and uncertainty. It is trained by minimising the Gaussian
       negative log-likelihood, <span class="math">L = ½[(t − μ)²/σ² + log σ² + log 2π]</span>, with Adam (learning rate 10<sup>−3</sup>),
       batch size 512, up to 1000 epochs and early stopping with patience 50, keeping the best weights.</p>
    <h3>Target</h3>
    <p>For each event a random offset t<sub>true</sub> ~ U(−1, 1)&nbsp;ns is added to all four pad times, and the network
       is trained to predict this offset from the shifted times and the pad amplitudes. The residual μ − t<sub>true</sub>
       therefore measures how precisely the network combines the pad information into one event time, including the
       charge-dependent (time-walk) and charge-sharing corrections it learns.</p>
    <h3>Training, validation and test</h3>
    <p>The data are shuffled and split with 10-fold cross-validation. In each fold, 90% of the events are used for
       training (of which 20% are held out for validation and early stopping) and the remaining 10% for testing, i.e.
       72% / 18% / 10% per fold. Each event is tested exactly once, by the model that did not see it in training, so the
       results use the full data set (346k events nominal, 271k with 10&nbsp;mV noise after removing incomplete events).</p>
    <p>Three scenarios are studied: (A1) training and testing on nominal data; (A2) the ten nominal models applied to
       the noisy data, with the noisy amplitudes corrected for a per-pad voltage offset, normalised with the nominal
       ranges, and combined as an ensemble (σ² = mean predicted variance + variance of the ten predictions);
       (A3) training and testing on the noisy data with the same procedure as A1.</p>
    <h3>Performance metrics</h3>
    <p>The timing resolution is the width of μ − t<sub>true</sub> from a double-Gaussian fit, and the calibration of the
       predicted uncertainty is checked with the pull (μ − t<sub>true</sub>)/σ, which should have unit width. Both are
       studied as a function of the total charge and of the track position; events with |pull| &gt; 5 are removed.</p>
  </div>
  <div class="card" style="margin-top:16px">
    <h3 style="margin-bottom:8px">At a glance</h3>
    <table class="facts"><tbody>
      <tr><td>Inputs</td><td>16 = 4 pads × (charge, e-peak amplitude, signal maximum, sigmoid time)</td></tr>
      <tr><td>Architecture</td><td>MLP, 5 × 32 ReLU, outputs μ and log σ²</td></tr>
      <tr><td>Loss</td><td>Gaussian negative log-likelihood (heteroscedastic)</td></tr>
      <tr><td>Optimiser</td><td>Adam, lr 10<sup>−3</sup>, batch 512, ≤ 1000 epochs, early stopping (patience 50)</td></tr>
      <tr><td>Target</td><td>t<sub>true</sub> ~ U(−1, 1) ns added to all pad times</td></tr>
      <tr><td>Validation</td><td>10-fold CV; per fold 72% train / 18% validation / 10% test</td></tr>
      <tr><td>Data</td><td>346,195 events (nominal), 271,320 events (10 mV noise)</td></tr>
    </tbody></table>
  </div>
  <p class="note">Note for the analysers: in the nominal training (A1) early stopping and checkpointing monitored the training
     loss; in the noisy training (A3) they monitored the validation loss. A2 uses t<sub>true</sub> ~ U(−0.5, 0.5) ns.</p>
</section>"""

seltoggle = ''.join(f'<button class="seg" data-sel="{k}">{v}</button>' for k, v in SEL_TITLE.items())
tabs = "".join(
    f'<button class="tab" data-target="{SHORT[a["name"]].lower()}">'
    f'<span class="dot" style="background:{STYLE[a["name"]][0]}"></span>{SHORT[a["name"]]} · {html.escape(a["label"])}</button>'
    for a in analyses if SEL(a) == "all") + '<button class="tab" data-target="net">Network &amp; method</button>'

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PICOSEC Micromegas — Timing network results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{ --bg:#f7f7f5; --card:#fff; --ink:#1b1d21; --muted:#5f6670; --line:#e3e3df; --accent:#1f5fa8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 Inter,system-ui,sans-serif; }}
  header {{ background:#14171c; color:#eef0f3; padding:44px 24px 36px; }}
  header .wrap, main {{ max-width:1240px; margin:0 auto; }}
  main {{ padding:0 24px 64px; }}
  .eyebrow {{ font:500 12px/1 'JetBrains Mono',monospace; letter-spacing:.12em; text-transform:uppercase; color:#8fb4e6; }}
  h1 {{ font-size:34px; line-height:1.15; margin:10px 0 12px; letter-spacing:-.01em; }}
  header p {{ color:#b9bfc8; max-width:820px; margin:0; }}
  h2 {{ font-size:22px; margin:44px 0 6px; letter-spacing:-.01em; }}
  h2 + p.lead {{ color:var(--muted); margin:0 0 18px; max-width:860px; }}
  h3 {{ font-size:16px; margin:0; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th, td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:middle; }}
  th {{ font-weight:600; color:var(--muted); font-size:12.5px; }}
  td.num, th.num {{ text-align:right; font-family:'JetBrains Mono',monospace; font-size:13px; }}
  tr:last-child td {{ border-bottom:none; }}
  .scroll {{ overflow-x:auto; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; vertical-align:middle; }}
  #chargePlot {{ width:100%; height:640px; }}
  .tabs {{ display:flex; gap:8px; flex-wrap:wrap; position:sticky; top:0; z-index:5; background:var(--bg); padding:12px 0; border-bottom:1px solid var(--line); }}
  .tab {{ font:500 14px Inter,sans-serif; border:1px solid var(--line); background:#fff; color:var(--ink); padding:9px 14px; border-radius:8px; cursor:pointer; }}
  .tab.active {{ background:var(--ink); color:#fff; border-color:var(--ink); }}
  .analysis {{ display:none; padding-top:18px; }}
  .prose p {{ margin:6px 0 14px; max-width:860px; }}
  .prose h3 {{ margin-top:6px; }}
  .math {{ font-family:'JetBrains Mono',monospace; font-size:.9em; white-space:nowrap; }}
  table.facts td:first-child {{ color:var(--muted); width:160px; font-weight:500; }}
  p.note {{ color:var(--muted); font-size:13px; margin:12px 2px 0; max-width:860px; }}
  #copyNet {{ font:500 13px Inter,sans-serif; border:1px solid var(--line); background:#fff; padding:6px 12px; border-radius:6px; cursor:pointer; }}
  tr.group td {{ background:#f1f2f0; font-weight:600; font-size:13px; color:var(--muted); padding:7px 12px; }}
  .segs {{ display:inline-flex; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#fff; }}
  .seg {{ font:500 14px Inter,sans-serif; border:none; background:#fff; color:var(--ink); padding:9px 14px; cursor:pointer; }}
  .seg + .seg {{ border-left:1px solid var(--line); }}
  .seg.active {{ background:#1f5fa8; color:#fff; }}
  .controls {{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; justify-content:space-between; }}
  .analysis.active {{ display:block; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-bottom:16px; }}
  .kpis div {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
  .kpis span {{ display:block; color:var(--muted); font-size:12.5px; }}
  .kpis b {{ font:600 22px 'JetBrains Mono',monospace; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  figure {{ margin:0; background:#fff; border:1px solid var(--line); border-radius:10px; padding:16px; }}
  figure.wide {{ grid-column:1 / -1; }}
  figure img {{ width:100%; height:auto; display:block; margin-top:10px; cursor:zoom-in; }}
  figcaption p {{ margin:2px 0 0; color:var(--muted); font-size:13px; }}
  ul.notes {{ margin:0; padding-left:20px; }}
  ul.notes li {{ margin:6px 0; }}
  code {{ font-family:'JetBrains Mono',monospace; font-size:.88em; background:#eef0f2; padding:1px 5px; border-radius:4px; }}
  #lightbox {{ position:fixed; inset:0; background:rgba(10,12,15,.88); display:none; align-items:center; justify-content:center; z-index:50; cursor:zoom-out; padding:24px; }}
  #lightbox img {{ max-width:100%; max-height:100%; background:#fff; border-radius:6px; }}
  footer {{ color:var(--muted); font-size:13px; margin-top:48px; border-top:1px solid var(--line); padding-top:16px; }}
  @media (max-width:860px) {{ .grid {{ grid-template-columns:1fr; }} .kpis {{ grid-template-columns:repeat(2,1fr); }} h1 {{ font-size:26px; }} #chargePlot {{ height:540px; }} }}
</style>
</head>
<body>
<header><div class="wrap">
  <div class="eyebrow">PICOSEC Micromegas · 4-pad timing network</div>
  <h1>Timing resolution with a heteroscedastic neural network</h1>
  <p>Out-of-fold (10-fold) predictions of the event time from per-pad charge, e-peak amplitude, signal maximum and fitted time.
     Three scenarios: nominal data, a network trained on nominal data evaluated on data with 10&nbsp;mV noise,
     and a network trained and evaluated on the noisy data. All times in picoseconds.</p>
</div></header>

<main>
  <h2>Summary</h2>
  <p class="lead">Double-Gaussian fit of the residual t<sub>network</sub> − t<sub>true</sub> and of the pull, after the cut |pull| &lt; 5.
     σ<sub>tot</sub> = √(f<sub>core</sub>σ<sub>core</sub>² + f<sub>tail</sub>σ<sub>tail</sub>²), with area fractions.</p>
  <div class="card scroll"><table>
    <thead><tr><th>Scenario</th><th class="num">Events</th><th class="num">σ<sub>core</sub> [ps]</th><th class="num">σ<sub>tail</sub> [ps]</th>
      <th class="num">f<sub>core</sub></th><th class="num">σ<sub>tot</sub> [ps]</th><th class="num">σ<sub>single</sub> [ps]</th><th class="num">Bias [ps]</th>
      <th class="num">σ(pull)</th><th class="num">μ(pull)</th><th class="num">Removed by pull cut</th></tr></thead>
    <tbody>{table}</tbody>
  </table></div>

  <h2>Bias and resolution vs charge</h2>
  <p class="lead">All three scenarios on common axes. Bias range fixed to ±10&nbsp;ps. Hover a point for its charge range and number of events;
     click a legend entry to hide a scenario. Pad-centre points use 1,000 events each (10,000 for all tracks).</p>
  <div class="segs" id="chargeSel">{seltoggle}</div>
  <div class="card" style="margin-top:10px"><div id="chargePlot"></div></div>

  <h2>Per-scenario results</h2>
  <p class="lead">Click any plot to enlarge it. Std heatmaps share one colour scale across scenarios, so the colours compare directly.</p>
  <nav class="tabs controls"><div style="display:flex;gap:8px;flex-wrap:wrap">{tabs}</div><div class="segs" id="secSel">{seltoggle}</div></nav>
  {''.join(sections)}
  {NET_SECTION}

  <h2>Method notes</h2>
  <div class="card"><ul class="notes">
    <li><b>Charge.</b> The network inputs are min-max normalised per column. The charge on this page is the real <code>qall_idx</code> value,
        taken from the ROOT file by the unique event id <code>srs</code> with the same threshold as training (<code>qall &lt; 1 → 0</code>);
        Q is the sum over the 4 pads. Inverting the normalisation gives the same values (max difference 2·10<sup>−5</sup>).</li>
    <li><b>Predictions.</b> Regenerated from the saved fold models (<code>best_model_fold*.h5</code>) with the training preprocessing.
        The shuffle and 10-fold split use <code>random_state=1312</code>, so every event is predicted by the model that did not see it.
        Train-nominal/test-noisy uses the 10-model ensemble with voltage-offset correction; σ<sub>network</sub> combines the mean
        predicted variance and the spread between models.</li>
    <li><b>Fits.</b> Two-pass Gaussian (second pass within ±5σ, binning with best R²). Total resolution and pulls use a double Gaussian
        with a common mean. Heatmap cells need at least 200 hits.</li>
    <li><b>Selection.</b> |pull| &lt; 5 in all scenarios. <i>Pad centres:</i> tracks with |hit_x − x<sub>c</sub>| ≤ 1.25&nbsp;mm and
        |hit_y − y<sub>c</sub>| ≤ 1.25&nbsp;mm for one of the six pad centres x<sub>c</sub> ∈ {{−1.002, 0, 1.002}}&nbsp;cm,
        y<sub>c</sub> ∈ {{−0.501, 0.501}}&nbsp;cm (square windows). With these statistics the double Gaussian can be unstable,
        so σ<sub>single</sub> (single Gaussian within ±5σ) is given as well. When the second Gaussian carries less than 10% of the area,
        the fit falls back to a single Gaussian (shown as f<sub>core</sub> = 100%, σ<sub>tail</sub> = –).</li>
  </ul></div>

  <footer>PICOSEC Micromegas Collaboration · AUTh · Generated with <code>picosec_analysis.py</code> and <code>build_page.py</code>.</footer>
</main>

<div id="lightbox"><img alt=""></div>

<script>
  // ---- Tabs (σενάριο) + επιλογή (όλα / κέντρα) ----
  const tabs = document.querySelectorAll('.tab');
  const secs = document.querySelectorAll('.analysis');
  let cur = {{ id:'a1', sel:'all' }};
  function show() {{
    const sid = cur.id === 'net' ? 'net' : cur.id + (cur.sel === 'c' ? 'c' : '');
    document.getElementById('secSel').style.visibility = cur.id === 'net' ? 'hidden' : 'visible';
    tabs.forEach(t => t.classList.toggle('active', t.dataset.target === cur.id));
    document.querySelectorAll('#secSel .seg').forEach(b => b.classList.toggle('active', b.dataset.sel === cur.sel));
    secs.forEach(s => s.classList.toggle('active', s.id === sid));
  }}
  function fromHash() {{
    const h = (location.hash || '').slice(1);
    if (/^a[123]c?$/.test(h)) cur = {{ id:h.slice(0,2), sel: h.endsWith('c') ? 'c' : 'all' }};
    else if (h === 'net') cur.id = 'net';
    show();
  }}
  const setHash = () => history.replaceState(null, '', '#' + (cur.id === 'net' ? 'net' : cur.id + (cur.sel === 'c' ? 'c' : '')));
  tabs.forEach(t => t.addEventListener('click', () => {{ cur.id = t.dataset.target; show(); setHash(); }}));
  document.querySelectorAll('#secSel .seg').forEach(b => b.addEventListener('click', () => {{ cur.sel = b.dataset.sel; show(); setHash(); }}));
  fromHash();
  window.addEventListener('hashchange', fromHash);

  // ---- Copy του κειμένου (για το paper) ----
  document.getElementById('copyNet').addEventListener('click', async e => {{
    try {{ await navigator.clipboard.writeText(document.getElementById('netText').innerText); e.target.textContent = 'Copied'; }}
    catch (err) {{ e.target.textContent = 'Copy failed — select the text'; }}
    setTimeout(() => e.target.textContent = 'Copy text', 2000);
  }});

  // ---- Lightbox ----
  const lb = document.getElementById('lightbox'), lbImg = lb.querySelector('img');
  document.querySelectorAll('figure img').forEach(im => im.addEventListener('click', () => {{ lbImg.src = im.src; lb.style.display = 'flex'; }}));
  lb.addEventListener('click', () => lb.style.display = 'none');
  document.addEventListener('keydown', e => {{ if (e.key === 'Escape') lb.style.display = 'none'; }});

  // ---- Interactive charge comparison ----
  const traces = {json.dumps(traces)};
  const axis = {{ gridcolor:'#ececea', zerolinecolor:'#9aa0a6', linecolor:'#9aa0a6', ticks:'outside', tickcolor:'#9aa0a6' }};
  const layout = {{
    font: {{ family:'Inter, sans-serif', size:13, color:'#1b1d21' }},
    paper_bgcolor:'#fff', plot_bgcolor:'#fff', margin: {{ l:70, r:20, t:10, b:60 }},
    xaxis: {{ ...axis, title:{{text:'Total charge Q'}}, anchor:'y2' }},
    yaxis: {{ ...axis, title:{{text:'Bias μ [ps]'}}, range:[-10,10], domain:[0.6,1], zeroline:true, zerolinewidth:1.5 }},
    yaxis2: {{ ...axis, title:{{text:'Resolution σ [ps]'}}, rangemode:'tozero', domain:[0,0.52] }},
    legend: {{ orientation:'h', x:0, y:-0.14 }},
    hoverlabel: {{ bgcolor:'#fff', bordercolor:'#c9ccd1', font: {{ family:'Inter, sans-serif' }} }},
  }};
  if (window.Plotly) {{
    Plotly.newPlot('chargePlot', traces, layout, {{ responsive:true, displaylogo:false }});
    const segs = document.querySelectorAll('#chargeSel .seg');
    const pick = sel => {{
      segs.forEach(b => b.classList.toggle('active', b.dataset.sel === sel));
      Plotly.restyle('chargePlot', {{ visible: traces.map(t => t.meta === sel) }});
    }};
    segs.forEach(b => b.addEventListener('click', () => pick(b.dataset.sel)));
    segs[0].classList.add('active');
  }} else {{
    document.getElementById('chargePlot').innerHTML = '{img("comparison_charge_analysis.png", "Bias and resolution vs charge")}';
  }}
</script>
</body>
</html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(page)
print(OUT, f"{os.path.getsize(OUT)/1e6:.1f} MB")
