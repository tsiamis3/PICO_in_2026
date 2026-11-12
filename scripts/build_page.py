"""
Φτιάχνει ένα αυτόνομο index.html (εικόνες ενσωματωμένες ως base64) με όλα τα αποτελέσματα.
Σκοτεινό θέμα (μαύρο / σκούρο κίτρινο), μία καρτέλα ανά είδος αποτελέσματος, με τις 3 αναλύσεις δίπλα-δίπλα.
Τρέξιμο:  python build_page.py   (μετά το picosec_analysis.py)
"""
import os
import json
import base64
import html
import pandas as pd

PLOTS = "plots"
OUT = "site/index.html"

analyses = json.load(open(os.path.join(PLOTS, "summary.json")))
BASE = ["A1_nominal", "A2_train_nominal_test_noisy", "A3_noisy_train"]
SHORT = {"A1_nominal": "A1", "A2_train_nominal_test_noisy": "A2", "A3_noisy_train": "A3"}
COLOR = {"A1": "#f2c230", "A2": "#ff6f59", "A3": "#4cc9b0"}       # ίδια με τα PNG
SYMBOL = {"A1": "circle", "A2": "square-open", "A3": "triangle-up-open"}
SEL_TITLE = {"all": "All tracks", "c": "Pad centres (±1.25 mm)"}
SUFFIX = {"all": "", "c": "_centres"}

by_name = {a["name"]: a for a in analyses}
A = {(SHORT[b], sel): by_name[b + SUFFIX[sel]] for b in BASE for sel in ("all", "c")}


def img(fname, alt):
    with open(os.path.join(PLOTS, fname), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img loading="lazy" src="data:image/png;base64,{b64}" alt="{html.escape(alt)}">'


def dl(base):
    """Links λήψης για τη λευκή έκδοση ενός plot (αρχεία στο plots_light/ του repo)."""
    return (f'<div class="dl">White version: <a href="plots_light/{base}.png" download>PNG</a>'
            f' · <a href="plots_light/{base}.pdf" download>PDF</a></div>')


def fmt_tail(v):
    return "–" if v != v else f"{v:.1f}"


# ======================= Overview: πίνακας σύνοψης =======================
rows = []
for sel in ("all", "c"):
    rows.append(f'<tr class="group"><td colspan="11">{SEL_TITLE[sel]}</td></tr>')
    for k in ("A1", "A2", "A3"):
        a = A[(k, sel)]
        r, p = a["resolution_ps"], a["pull"]
        rows.append(f"""<tr>
      <td><span class="dot" style="background:{COLOR[k]}"></span><b>{k}</b>&nbsp;{html.escape(a['label'])}</td>
      <td class="num">{a['n_events']:,}</td>
      <td class="num">{r['sigma_core']:.1f}</td>
      <td class="num">{fmt_tail(r['sigma_tail'])}</td>
      <td class="num">{100*r['frac_core']:.0f}%</td>
      <td class="num hi">{r['sigma_tot']:.1f}</td>
      <td class="num">{r['sigma_single']:.1f}</td>
      <td class="num">{r['mu']:+.1f}</td>
      <td class="num">{p['sigma_tot']:.2f}</td>
      <td class="num">{p['mu']:+.3f}</td>
      <td class="num">{100*a['frac_removed_by_pull_cut']:.2f}%</td>
    </tr>""")
summary_table = "\n".join(rows)

kpi_cards = "".join(f"""<div class="kpi" style="--c:{COLOR[k]}">
    <div class="kpi-head"><span class="dot" style="background:{COLOR[k]}"></span>{k} · {html.escape(A[(k,'all')]['label'])}</div>
    <div class="kpi-val">{A[(k,'all')]['resolution_ps']['sigma_tot']:.1f}<small> ps</small></div>
    <div class="kpi-sub">σ<sub>tot</sub> all tracks · <b>{A[(k,'c')]['resolution_ps']['sigma_tot']:.1f} ps</b> at pad centres · σ(pull) {A[(k,'all')]['pull']['sigma_tot']:.2f}</div>
  </div>""" for k in ("A1", "A2", "A3"))

# ======================= Δεδομένα για το διαδραστικό charge plot =======================
traces = []
for (k, sel), a in A.items():
    fr = pd.read_csv(os.path.join(PLOTS, f"{a['name']}_charge_analysis.csv"))
    c = COLOR[k]
    for yaxis, y, err in (("y", "Mu", "Mu_error"), ("y2", "Sigma", "Sigma_error")):
        traces.append(dict(
            x=fr["Q"].round(3).tolist(), y=fr[y].round(3).tolist(),
            error_y=dict(type="data", array=fr[err].round(3).tolist(), thickness=1.2, width=3, color=c),
            customdata=fr[["Q_min", "Q_max", "N"]].round(2).values.tolist(),
            mode="markers", marker=dict(color=c, symbol=SYMBOL[k], size=7, line=dict(color=c, width=1.5)),
            name=f"{k} · {a['label']}", legendgroup=k, showlegend=(yaxis == "y2"), yaxis=yaxis,
            meta=sel, visible=(sel == "all"),
            hovertemplate=("<b>%{fullData.name}</b><br>Q = %{x:.1f} (range %{customdata[0]:.1f}–%{customdata[1]:.1f})"
                           "<br>" + ("bias" if y == "Mu" else "σ") + " = %{y:.2f} ± %{error_y.array:.2f} ps"
                           "<br>N = %{customdata[2]:,}<extra></extra>"),
        ))

# ======================= Καρτέλες αποτελεσμάτων =======================
# (id, τίτλος καρτέλας, αρχείο, περιγραφή, wide: ένα plot ανά γραμμή, kpi: τι νούμερο δείχνει η κάρτα)
RESULT_TABS = [
    ("resolution", "Total resolution", "1_total_resolution",
     "Double-Gaussian fit (common mean) of t<sub>network</sub> − t<sub>true</sub>. "
     "σ<sub>tot</sub> = √(f<sub>core</sub>σ<sub>core</sub>² + f<sub>tail</sub>σ<sub>tail</sub>²) with area fractions.",
     False, lambda a: f"σ<sub>tot</sub> = {a['resolution_ps']['sigma_tot']:.1f} ps · bias {a['resolution_ps']['mu']:+.1f} ps"),
    ("pulls", "Pulls", "2_pulls",
     "(t<sub>network</sub> − t<sub>true</sub>) / σ<sub>network</sub>. A well-calibrated uncertainty gives unit width.",
     False, lambda a: f"σ(pull) = {a['pull']['sigma_tot']:.2f} · μ(pull) {a['pull']['mu']:+.3f}"),
    ("charge", "Charge analysis", "3_charge_analysis",
     "Events sorted by the total charge Q; Gaussian fit of t<sub>network</sub> − t<sub>true</sub> in consecutive groups "
     "of 10,000 events (all tracks) or 1,000 events (pad centres). Bias axis fixed to ±10 ps.",
     False, lambda a: f"{a['events_per_charge_point']:,} events / point"),
    ("heat-dt", "Heatmaps: Δt", "4_heatmap_tt",
     "Bias and std of t<sub>network</sub> − t<sub>true</sub> per (hit_x, hit_y) cell, or per pad centre for the pad-centre "
     "selection (≥ 200 hits). Bias scale ±10 ps; one std scale for all scenarios.",
     True, lambda a: f"{a['n_events']:,} events"),
    ("heat-pull", "Heatmaps: pulls", "5_heatmap_pull",
     "Bias and std of the pull per cell. Diverging scales centred on 0 (bias) and 1 (std).",
     True, lambda a: f"{a['n_events']:,} events"),
]

result_sections = []
for tid, title, kind, desc, wide, kpi in RESULT_TABS:
    blocks = []
    for sel in ("all", "c"):
        figs = []
        for k in ("A1", "A2", "A3"):
            a = A[(k, sel)]
            figs.append(f"""<figure>
              <figcaption><span class="dot" style="background:{COLOR[k]}"></span><b>{k}</b>&nbsp;{html.escape(a['label'])}
                <span class="fig-kpi">{kpi(a)}</span></figcaption>
              {img(f"{a['name']}_{kind}.png", f"{title} — {k} — {SEL_TITLE[sel]}")}
              {dl(f"{a['name']}_{kind}")}
            </figure>""")
        blocks.append(f'<div class="figs {"wide" if wide else ""}" data-sel="{sel}">{"".join(figs)}</div>')
    result_sections.append(f"""<section class="tabpane" id="{tid}">
      <div class="pane-head"><h2>{title}</h2><p>{desc}</p></div>
      {''.join(blocks)}
    </section>""")

# ======================= Network & method =======================
NET_SECTION = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "net_section.html"), encoding="utf-8").read()

TABS = [("overview", "Overview"), ("compare", "Bias &amp; σ vs charge")] + \
       [(t[0], t[1]) for t in RESULT_TABS] + [("net", "Network &amp; method")]
TABS_WITH_SEL = ["compare"] + [t[0] for t in RESULT_TABS]
nav = "".join(f'<button class="tab" data-target="{tid}">{name}</button>' for tid, name in TABS)
seltoggle = "".join(f'<button class="seg" data-sel="{k}">{v}</button>' for k, v in SEL_TITLE.items())

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>PICOSEC Micromegas — Timing network results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg:#0e0e0c; --panel:#16150f; --panel2:#1d1b14; --line:#2c2a22; --ink:#e9e4d4; --muted:#9d977f;
    --gold:#d4a017; --gold-hi:#f2c230; --gold-dim:#8a6a12;
  }}
  * {{ box-sizing:border-box; }}
  html {{ background:var(--bg); }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 Inter,system-ui,sans-serif; }}
  a {{ color:var(--gold-hi); }}
  header {{ border-bottom:1px solid var(--line); background:
           radial-gradient(1200px 300px at 10% -20%, rgba(212,160,23,.18), transparent 60%), var(--bg); }}
  header .wrap {{ max-width:1320px; margin:0 auto; padding:40px 24px 30px; position:relative; }}
  header .zipbtn {{ position:absolute; right:24px; top:40px; }}
  @media (max-width:900px) {{ header .zipbtn {{ position:static; display:inline-block; margin:0 0 12px; }} }}
  .eyebrow {{ font:500 12px/1 'JetBrains Mono',monospace; letter-spacing:.14em; text-transform:uppercase; color:var(--gold); }}
  h1 {{ font-size:34px; line-height:1.15; margin:12px 0 10px; letter-spacing:-.015em; }}
  h1 em {{ font-style:normal; color:var(--gold-hi); }}
  header p {{ color:var(--muted); max-width:860px; margin:0; }}
  nav.bar {{ position:sticky; top:0; z-index:10; background:rgba(14,14,12,.94); backdrop-filter:blur(6px); border-bottom:1px solid var(--line); }}
  nav.bar .wrap {{ max-width:1320px; margin:0 auto; padding:10px 24px; display:flex; gap:12px; align-items:center; justify-content:space-between; flex-wrap:wrap; }}
  .tabs {{ display:flex; gap:4px; flex-wrap:wrap; }}
  .tab {{ font:500 14px Inter,sans-serif; border:1px solid transparent; background:transparent; color:var(--muted);
          padding:8px 12px; border-radius:7px; cursor:pointer; }}
  .tab:hover {{ color:var(--ink); background:var(--panel); }}
  .tab.active {{ color:#0e0e0c; background:var(--gold); }}
  .segs {{ display:inline-flex; border:1px solid var(--line); border-radius:7px; overflow:hidden; }}
  .seg {{ font:500 13px Inter,sans-serif; border:none; background:var(--panel); color:var(--muted); padding:8px 12px; cursor:pointer; }}
  .seg + .seg {{ border-left:1px solid var(--line); }}
  .seg.active {{ background:var(--panel2); color:var(--gold-hi); box-shadow:inset 0 -2px 0 var(--gold); }}
  main {{ max-width:1320px; margin:0 auto; padding:8px 24px 64px; }}
  .tabpane {{ display:none; }}
  .tabpane.active {{ display:block; }}
  .pane-head h2 {{ font-size:22px; margin:26px 0 4px; letter-spacing:-.01em; }}
  .pane-head p {{ color:var(--muted); margin:0 0 18px; max-width:900px; }}
  h3 {{ font-size:16px; margin:0 0 6px; color:var(--gold-hi); }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:18px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; margin-bottom:16px; }}
  .kpi {{ background:var(--panel); border:1px solid var(--line); border-top:3px solid var(--c); border-radius:10px; padding:14px 16px; }}
  .kpi-head {{ color:var(--muted); font-size:13px; }}
  .kpi-val {{ font:600 30px 'JetBrains Mono',monospace; color:var(--ink); margin:4px 0; }}
  .kpi-val small {{ font-size:14px; color:var(--muted); }}
  .kpi-sub {{ color:var(--muted); font-size:13px; }}
  .kpi-sub b {{ color:var(--ink); font-weight:600; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th, td {{ padding:9px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:middle; }}
  th {{ font-weight:600; color:var(--muted); font-size:12.5px; }}
  td.num, th.num {{ text-align:right; font-family:'JetBrains Mono',monospace; font-size:13px; }}
  td.hi {{ color:var(--gold-hi); font-weight:600; }}
  tr:last-child td {{ border-bottom:none; }}
  tr.group td {{ background:var(--panel2); color:var(--gold); font-weight:600; font-size:12.5px; letter-spacing:.04em; text-transform:uppercase; padding:7px 12px; }}
  .scroll {{ overflow-x:auto; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; vertical-align:middle; }}
  .figs {{ display:none; grid-template-columns:repeat(3,1fr); gap:14px; }}
  .figs.wide {{ grid-template-columns:1fr; }}
  .figs.wide figure {{ max-width:1000px; margin:0 auto; width:100%; }}
  .figs.on {{ display:grid; }}
  figure {{ margin:0; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 12px 10px; }}
  figcaption {{ font-size:14px; display:flex; align-items:center; flex-wrap:wrap; gap:2px 0; }}
  .fig-kpi {{ margin-left:auto; font:500 12.5px 'JetBrains Mono',monospace; color:var(--gold-hi); }}
  figure img {{ width:100%; height:auto; display:block; margin-top:10px; border-radius:4px; cursor:zoom-in; }}
  #chargePlot {{ width:100%; height:660px; }}
  .two {{ display:grid; grid-template-columns:1.25fr 1fr; gap:16px; margin-top:16px; }}
  ul.notes {{ margin:0; padding-left:20px; color:var(--ink); }}
  ul.notes li {{ margin:7px 0; }}
  ul.notes b {{ color:var(--gold-hi); }}
  code {{ font-family:'JetBrains Mono',monospace; font-size:.86em; background:var(--panel2); border:1px solid var(--line); padding:0 5px; border-radius:4px; color:var(--ink); }}
  .prose p {{ margin:6px 0 14px; max-width:880px; }}
  .math {{ font-family:'JetBrains Mono',monospace; font-size:.9em; white-space:nowrap; color:var(--gold-hi); }}
  table.facts td:first-child {{ color:var(--muted); width:160px; font-weight:500; }}
  p.note {{ color:var(--muted); font-size:13px; margin:12px 2px 0; max-width:880px; }}
  #copyNet {{ font:500 13px Inter,sans-serif; border:1px solid var(--gold-dim); background:transparent; color:var(--gold-hi); padding:6px 12px; border-radius:6px; cursor:pointer; }}
  #copyNet:hover {{ background:rgba(212,160,23,.12); }}
  #net {{ padding-top:22px; }}
  #lightbox {{ position:fixed; inset:0; background:rgba(0,0,0,.92); display:none; align-items:center; justify-content:center; z-index:50; cursor:zoom-out; padding:24px; }}
  #lightbox img {{ max-width:100%; max-height:100%; border-radius:6px; border:1px solid var(--line); }}
  footer {{ color:var(--muted); font-size:13px; margin-top:48px; border-top:1px solid var(--line); padding-top:16px; }}
  .hidden {{ visibility:hidden; }}
  .dl {{ font-size:12.5px; color:var(--muted); margin-top:8px; }}
  .dl a {{ color:var(--gold-hi); text-decoration:none; border-bottom:1px dotted var(--gold-dim); }}
  .dl a:hover {{ border-bottom-style:solid; }}
  .zipbtn {{ font:500 13px Inter,sans-serif; color:#0e0e0c; background:var(--gold); padding:8px 12px; border-radius:7px; text-decoration:none; white-space:nowrap; }}
  .zipbtn:hover {{ background:var(--gold-hi); }}
  @media (max-width:1100px) {{ .figs {{ grid-template-columns:1fr 1fr; }} .two {{ grid-template-columns:1fr; }} }}
  @media (max-width:720px) {{ .figs {{ grid-template-columns:1fr; }} h1 {{ font-size:26px; }} #chargePlot {{ height:560px; }} }}
</style>
</head>
<body>
<header><div class="wrap">
  <div class="eyebrow">PICOSEC Micromegas · 4-pad timing network</div>
  <h1>Timing resolution with a <em>heteroscedastic</em> neural network</h1>
  <a class="zipbtn" href="plots_light.zip" download title="All plots, white background, PNG + PDF">⬇ All plots, white (zip)</a>
  <p>Out-of-fold (10-fold) predictions of the event time from per-pad charge, e-peak amplitude, signal maximum and fitted time,
     for nominal data and for data with 10&nbsp;mV noise. All times in picoseconds.</p>
</div></header>

<nav class="bar"><div class="wrap">
  <div class="tabs">{nav}</div>
  <div class="segs" id="selToggle">{seltoggle}</div>
</div></nav>

<main>
  <section class="tabpane" id="overview">
    <div class="pane-head"><h2>Overview</h2>
      <p>Three scenarios — <b style="color:{COLOR['A1']}">A1</b> train and test on nominal data,
         <b style="color:{COLOR['A2']}">A2</b> train on nominal and test on noisy data,
         <b style="color:{COLOR['A3']}">A3</b> train and test on noisy data — for all tracks and for tracks at the pad centres.</p></div>
    <div class="kpis">{kpi_cards}</div>
    <div class="card scroll"><table>
      <thead><tr><th>Scenario</th><th class="num">Events</th><th class="num">σ<sub>core</sub> [ps]</th><th class="num">σ<sub>tail</sub> [ps]</th>
        <th class="num">f<sub>core</sub></th><th class="num">σ<sub>tot</sub> [ps]</th><th class="num">σ<sub>single</sub> [ps]</th><th class="num">Bias [ps]</th>
        <th class="num">σ(pull)</th><th class="num">μ(pull)</th><th class="num">Removed by pull cut</th></tr></thead>
      <tbody>{summary_table}</tbody>
    </table></div>
    <div class="two">
      <figure><figcaption><h3 style="margin:0">Pad-centre selection</h3></figcaption>
        {img("selection_pad_centres.png", "Pad centre selection")}{dl("selection_pad_centres")}</figure>
      <div class="card"><h3>Method notes</h3><ul class="notes">
        <li><b>Charge.</b> Real <code>qall_idx</code> values from the ROOT file (joined on the event id <code>srs</code>,
            same threshold as training: <code>qall &lt; 1 → 0</code>); Q is the sum over the 4 pads. Inverting the min-max
            normalisation gives the same values (max difference 2·10<sup>−5</sup>).</li>
        <li><b>Predictions.</b> Regenerated from the saved fold models; shuffle and 10-fold split with
            <code>random_state=1312</code>, so every event is predicted by the model that did not see it.</li>
        <li><b>Fits.</b> Two-pass Gaussian (second pass in ±5σ, binning with best R²); double Gaussian for resolution and pulls,
            falling back to a single Gaussian when a component carries &lt; 10% of the area.</li>
        <li><b>Selections.</b> |pull| &lt; 5 everywhere. Pad centres: |Δx|, |Δy| ≤ 1.25&nbsp;mm around
            x<sub>c</sub> ∈ {{−1.002, 0, 1.002}}&nbsp;cm, y<sub>c</sub> ∈ {{−0.501, 0.501}}&nbsp;cm (~4% of tracks).</li>
      </ul></div>
    </div>
  </section>

  <section class="tabpane" id="compare">
    <div class="pane-head"><h2>Bias and resolution vs charge</h2>
      <p>All three scenarios on common axes; bias fixed to ±10&nbsp;ps. Hover a point for its charge range and number of events,
         click a legend entry to hide a scenario. The selection switch in the bar changes the track selection.</p></div>
    <div class="card"><div id="chargePlot"></div></div>
    <div class="dl" style="margin-top:8px">Static white version: all tracks
      <a href="plots_light/comparison_charge_analysis.png" download>PNG</a> · <a href="plots_light/comparison_charge_analysis.pdf" download>PDF</a>
      &nbsp;|&nbsp; pad centres
      <a href="plots_light/comparison_charge_analysis_centres.png" download>PNG</a> · <a href="plots_light/comparison_charge_analysis_centres.pdf" download>PDF</a></div>
  </section>

  {''.join(result_sections)}

  <section class="tabpane" id="net-pane">
    <div class="pane-head"><h2>Network &amp; method</h2>
      <p>Short description of the network and of the training, validation and test procedure, for use in the paper.</p></div>
    <figure style="margin-bottom:16px"><figcaption><h3 style="margin:0">Architecture</h3>
      <span class="fig-kpi">16 → 5 × 32 ReLU → (μ, log σ²) · 4,834 parameters</span></figcaption>
      {img("network_architecture.png", "Network architecture")}{dl("network_architecture")}</figure>
    {NET_SECTION}
  </section>

  <footer>PICOSEC Micromegas Collaboration · AUTh · Generated with <code>picosec_analysis.py</code> and <code>build_page.py</code>.</footer>
</main>

<div id="lightbox"><img alt=""></div>

<script>
  const SEL_TABS = {json.dumps(TABS_WITH_SEL)};
  const tabs = document.querySelectorAll('.tab');
  const panes = document.querySelectorAll('.tabpane');
  const segs = document.querySelectorAll('#selToggle .seg');
  let cur = {{ tab:'overview', sel:'all' }};

  function render() {{
    tabs.forEach(t => t.classList.toggle('active', t.dataset.target === cur.tab));
    panes.forEach(p => p.classList.toggle('active', p.id === cur.tab || (cur.tab === 'net' && p.id === 'net-pane')));
    segs.forEach(b => b.classList.toggle('active', b.dataset.sel === cur.sel));
    document.getElementById('selToggle').classList.toggle('hidden', !SEL_TABS.includes(cur.tab));
    document.querySelectorAll('.figs').forEach(f => f.classList.toggle('on', f.dataset.sel === cur.sel));
    if (window.Plotly && cur.tab === 'compare') {{
      Plotly.restyle('chargePlot', {{ visible: traces.map(t => t.meta === cur.sel) }});
      Plotly.Plots.resize('chargePlot');
    }}
  }}
  function fromHash() {{
    const h = (location.hash || '').slice(1);
    const m = h.match(/^([a-z-]+?)(-c)?$/);
    if (m && [...tabs].some(t => t.dataset.target === m[1])) {{ cur.tab = m[1]; cur.sel = m[2] ? 'c' : 'all'; }}
    render();
  }}
  function setHash() {{
    history.replaceState(null, '', '#' + cur.tab + (SEL_TABS.includes(cur.tab) && cur.sel === 'c' ? '-c' : ''));
  }}
  tabs.forEach(t => t.addEventListener('click', () => {{ cur.tab = t.dataset.target; render(); setHash(); window.scrollTo({{top:0}}); }}));
  segs.forEach(b => b.addEventListener('click', () => {{ cur.sel = b.dataset.sel; render(); setHash(); }}));
  window.addEventListener('hashchange', fromHash);

  // ---- Copy του κειμένου του network (για το paper) ----
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

  // ---- Διαδραστικό charge plot ----
  const traces = {json.dumps(traces)};
  const axis = {{ gridcolor:'#2c2a22', zerolinecolor:'#8a8470', linecolor:'#5a5648', ticks:'outside', tickcolor:'#5a5648',
                 tickfont: {{ color:'#9d977f' }} }};
  const layout = {{
    font: {{ family:'Inter, sans-serif', size:13, color:'#e9e4d4' }},
    paper_bgcolor:'#16150f', plot_bgcolor:'#16150f', margin: {{ l:70, r:20, t:10, b:60 }},
    xaxis: {{ ...axis, title: {{ text:'Total charge Q' }}, anchor:'y2' }},
    yaxis: {{ ...axis, title: {{ text:'Bias μ [ps]' }}, range:[-10,10], domain:[0.6,1], zeroline:true, zerolinewidth:1.5 }},
    yaxis2: {{ ...axis, title: {{ text:'Resolution σ [ps]' }}, rangemode:'tozero', domain:[0,0.52] }},
    legend: {{ orientation:'h', x:0, y:-0.14, font: {{ color:'#e9e4d4' }} }},
    hoverlabel: {{ bgcolor:'#1d1b14', bordercolor:'#d4a017', font: {{ family:'Inter, sans-serif', color:'#e9e4d4' }} }},
  }};
  if (window.Plotly) {{
    Plotly.newPlot('chargePlot', traces, layout, {{ responsive:true, displaylogo:false }});
  }} else {{
    document.getElementById('chargePlot').style.height = 'auto';
    document.getElementById('chargePlot').innerHTML = '{img("comparison_charge_analysis.png", "Bias and resolution vs charge")}';
  }}
  fromHash();
</script>
</body>
</html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(page)
print(OUT, f"{os.path.getsize(OUT)/1e6:.1f} MB")
