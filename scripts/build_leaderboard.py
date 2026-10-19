#!/usr/bin/env python3
"""Build a self-contained leaderboard page from reconstructions/.

Scans every group directory under `reconstructions/` (e.g. OCMR-lowacc),
reads its `summary_table.csv` -- or, when absent, rebuilds the numbers from
the per-split `metrics_by_method.csv` files -- and writes `leaderboard.html`
next to the repository root. The page embeds the data as JSON, so it works
offline from `file://` with no server and no external assets.

Run it after any metric refresh:

    python scripts/build_leaderboard.py

Inputs (all produced by scripts/calc_metric.py):
    reconstructions/<group>/summary_table.csv          # wide table, one row per method
    reconstructions/<group>/<split>/metrics_by_method.csv
    reconstructions/<group>/<split>/metrics_summary.json   # per-sample counts
Outputs:
    leaderboard.html
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECON = ROOT / "reconstructions"
WEIGHTS = ROOT / "weights"
OUT = ROOT / "leaderboard.html"

COL_RE = re.compile(r"^(?P<metric>PSNR|SSIM|HFEN)@(?P<acc>\d+)X$")
PM_RE = re.compile(r"^\s*([-\d.]+)\s*±\s*([-\d.]+)\s*$")

# metric metadata: display name, unit, sort direction (True = higher is better)
METRICS = {
    "PSNR": {"label": "PSNR", "unit": "dB", "higher_better": True},
    "SSIM": {"label": "SSIM", "unit": "%", "higher_better": True},
    "HFEN": {"label": "HFEN", "unit": "\u00d71e-3", "higher_better": False},
}


def parse_pm(text: str) -> tuple[float, float]:
    m = PM_RE.match(str(text))
    if not m:
        raise ValueError(f"cannot parse 'mean \u00b1 std' from {text!r}")
    return float(m.group(1)), float(m.group(2))


def read_group(group_dir: Path) -> dict | None:
    """Return {'name','n','accs','methods':{m:{'MOTRIC@acc':(mean,std)}},'ckpt'}."""
    # 1) preferred: the wide summary table
    wide = group_dir / "summary_table.csv"
    cells: dict[str, dict[tuple[str, str], tuple[float, float]]] = {}
    accs: list[str] = []
    if wide.exists():
        for row in csv.DictReader(wide.open(encoding="utf-8")):
            method = row["method"].strip()
            bucket = cells.setdefault(method, {})
            for col, val in row.items():
                if col == "method" or not val or not val.strip():
                    continue
                m = COL_RE.match(col.strip())
                if not m:
                    continue
                metric, acc = m.group("metric"), m.group("acc")
                if acc not in accs:
                    accs.append(acc)
                bucket[(metric, acc)] = parse_pm(val)
    else:
        # 2) fallback: aggregate the per-split summaries ourselves
        for split_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            csv_path = split_dir / "metrics_by_method.csv"
            if not csv_path.exists():
                continue
            am = re.search(r"(\d+)$", split_dir.name)
            acc = str(int(am.group(1))) if am else split_dir.name
            if acc not in accs:
                accs.append(acc)
            for row in csv.DictReader(csv_path.open(encoding="utf-8")):
                method = row["recon_file"].replace("_reconstructions.h5", "").replace("_", "-")
                bucket = cells.setdefault(method, {})
                for metric in ("PSNR", "SSIM", "HFEN"):
                    key = {"PSNR": "PSNR (dB)", "SSIM": "SSIM (%)", "HFEN": "HFEN (x1e-3)"}[metric]
                    bucket[(metric, acc)] = parse_pm(row[key])

    if not cells:
        return None

    # per-sample count, straight from any split's summary json
    n = None
    for js in sorted(group_dir.glob("*/metrics_summary.json")):
        try:
            d = json.load(js.open())
            # total_samples counts one row per (method, sample); the split's
            # own sample count is that divided by the number of methods
            n = d["total_samples"] // max(len(d.get("by_recon_file", {})), 1)
            break
        except Exception:
            continue

    def acc_key(a: str) -> int:
        return int(a) if a.isdigit() else 1 << 30

    return {
        "name": group_dir.name,
        "n": n,
        "accs": sorted(accs, key=acc_key),
        "methods": {
            m: {f"{k[0]}@{k[1]}": v for k, v in d.items()} for m, d in cells.items()
        },
        "ckpt": {m: (WEIGHTS / group_dir.name / f"{m}.ckpt").exists() for m in cells},
    }


# reading order the page shows: OCMR-lowacc -> OCMR-highacc -> CMRxRecon
# (directory order would be alphabetical, which puts CMRxRecon first)
GROUP_ORDER = {"OCMR-lowacc": 0, "OCMR-highacc": 1, "CMRxRecon": 2}


def build() -> str:
    groups = [rg for g in sorted(p for p in RECON.iterdir() if p.is_dir())
              if (rg := read_group(g)) is not None]
    groups.sort(key=lambda g: (GROUP_ORDER.get(g["name"], 99), g["name"]))
    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "metrics": METRICS,
        "groups": groups,
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    # placeholder substitution avoids f-string brace clashes with CSS/JS
    html = TEMPLATE
    for token, value in (("__DATA__", data_json),
                         ("__GENERATED__", payload["generated"]),
                         ("__NGROUPS__", str(len(groups))),
                         ("__NMETHODS__", str(len({m for g in groups for m in g["methods"]}))),
                         ("__NSPLITS__", str(sum(len(g["accs"]) for g in groups)))):
        html = html.replace(token, value)
    OUT.write_text(html, encoding="utf-8")
    return data_json


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Benchmark leaderboard for MRI reconstruction models: PSNR, SSIM and HFEN across undersampling accelerations, recomputed from per-sample reconstructions.">
<title>cmr-recon-arena · MRI Reconstruction Leaderboard</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --fg: #16181d; --muted: #69707d;
    --line: #e6e8ec; --accent: #3b5bdb; --accent-soft: #eef1ff;
    --best-bg: #ecf8f0; --best-fg: #177245;
    --track: #eceef2; --fill: #7c96f2;
    --gold: #e8a33d; --silver: #9aa4b2; --bronze: #c97b4a; --plain: #b6bcc7;
    --shadow: 0 1px 2px rgba(16, 20, 30, .05), 0 8px 24px -12px rgba(16, 20, 30, .12);
    --radius: 12px;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #101216; --panel: #171a20; --fg: #e9ebef; --muted: #98a0ad;
      --line: #262a32; --accent: #7b93ff; --accent-soft: #1e2436;
      --best-bg: #14251c; --best-fg: #63d39a;
      --track: #22262e; --fill: #5f7eea;
      --gold: #f0b45f; --silver: #aab3c0; --bronze: #d9916a; --plain: #5b6270;
      --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 10px 30px -14px rgba(0, 0, 0, .6);
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 14.5px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
          "Helvetica Neue", Arial, "Noto Sans SC", sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1160px; margin: 0 auto; padding: 44px 28px 72px; }

  /* ---------- header ---------- */
  .eyebrow { font-size: 12px; font-weight: 600; letter-spacing: .09em;
             text-transform: uppercase; color: var(--accent); margin-bottom: 10px; }
  h1 { font-size: 30px; line-height: 1.2; margin: 0 0 10px; letter-spacing: -.02em; }
  .lede { color: var(--muted); max-width: 66ch; margin: 0 0 18px; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip { background: var(--panel); border: 1px solid var(--line); border-radius: 999px;
          padding: 5px 13px; font-size: 12.5px; color: var(--muted); box-shadow: var(--shadow); }
  .chip b { color: var(--fg); font-weight: 600; }

  /* ---------- controls ---------- */
  .controls { display: flex; gap: 22px; flex-wrap: wrap; align-items: flex-end;
              margin: 34px 0 18px; }
  .ctl-label { font-size: 11.5px; font-weight: 600; letter-spacing: .07em;
               text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
  .tabs { display: flex; gap: 10px; flex-wrap: wrap; }
  .tab { background: var(--panel); border: 1.5px solid var(--line); border-radius: var(--radius);
         padding: 10px 18px; cursor: pointer; text-align: left; min-width: 168px;
         transition: border-color .15s, transform .15s, box-shadow .15s; box-shadow: var(--shadow); }
  .tab:hover { transform: translateY(-1px); }
  .tab .t-name { font-weight: 650; font-size: 14.5px; color: var(--fg); display: block; }
  .tab .t-meta { font-size: 12px; color: var(--muted); }
  .tab.on { border-color: var(--accent); background: var(--accent-soft); }
  .pills { display: flex; gap: 8px; }
  .pill { background: var(--panel); border: 1.5px solid var(--line); border-radius: 999px;
          padding: 8px 17px; cursor: pointer; font: inherit; font-size: 13.5px;
          color: var(--muted); transition: all .15s; box-shadow: var(--shadow); }
  .pill:hover { color: var(--fg); }
  .pill.on { background: var(--accent); border-color: var(--accent); color: #fff; }

  /* ---------- table ---------- */
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
          box-shadow: var(--shadow); overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; min-width: 720px; }
  th, td { padding: 12px 14px; text-align: right; white-space: nowrap; }
  th:first-child, td:first-child, th.mcol, td.mcol { text-align: left; }
  thead th { font-size: 12px; font-weight: 600; letter-spacing: .04em; color: var(--muted);
             border-bottom: 1px solid var(--line); user-select: none; }
  thead th.sortable { cursor: pointer; }
  thead th.sortable:hover { color: var(--fg); }
  thead th.act { color: var(--accent); }
  thead th .arr { font-size: 10px; }
  tbody tr { transition: background .12s; }
  tbody tr:hover { background: color-mix(in srgb, var(--accent-soft) 45%, transparent); }
  tbody td { border-bottom: 1px solid var(--line); }
  tbody tr:last-child td { border-bottom: 0; }

  .pos { width: 40px; }
  .medal { display: inline-flex; align-items: center; justify-content: center;
           width: 24px; height: 24px; border-radius: 50%; font-size: 12px; font-weight: 700;
           background: var(--track); color: var(--muted); }
  .medal.g1 { background: var(--gold); color: #4a2c06; }
  .medal.g2 { background: var(--silver); color: #333a45; }
  .medal.g3 { background: var(--bronze); color: #43200a; }
  .mname { font-weight: 650; letter-spacing: -.01em; }
  .mck { display: inline-block; margin-left: 8px; font-size: 10.5px; font-weight: 600;
         letter-spacing: .04em; color: var(--best-fg); background: var(--best-bg);
         border: 1px solid color-mix(in srgb, var(--best-fg) 35%, transparent);
         padding: 1.5px 7px; border-radius: 999px; vertical-align: 1.5px; }
  .mck.no { color: var(--muted); background: var(--track); border-color: transparent; }

  /* value cell: number block on top, track bar strictly below it */
  .cell { min-width: 128px; }
  .cell .v { font-variant-numeric: tabular-nums; font-weight: 650; font-size: 14.5px;
             letter-spacing: -.01em; }
  .cell .s { font-variant-numeric: tabular-nums; font-size: 11.5px; color: var(--muted);
             margin-top: -2px; }
  .cell .track { margin-top: 7px; height: 5px; border-radius: 999px;
                 background: var(--track); overflow: hidden; }
  .cell .bar { display: block; height: 100%; border-radius: 999px; background: var(--fill);
               width: 0; animation: grow .5s cubic-bezier(.2, .7, .3, 1) forwards; }
  @keyframes grow { from { width: 0; } }
  .cell.sorted:not(.best) { background: color-mix(in srgb, var(--accent-soft) 70%, transparent); }
  .cell.best { background: var(--best-bg); }
  .cell.best .v { color: var(--best-fg); }
  .cell.best .bar { background: var(--best-fg); }
  .cell.best .s { color: color-mix(in srgb, var(--best-fg) 70%, var(--muted)); }

  .spark { min-width: 112px; }
  .spark svg { vertical-align: middle; }
  .sumcol { background: color-mix(in srgb, var(--accent-soft) 55%, transparent); }
  .numcol { font-variant-numeric: tabular-nums; font-weight: 600; }
  .rankcol { color: var(--muted); font-weight: 600; }

  /* ---------- footer ---------- */
  .notes { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
           gap: 14px; margin-top: 26px; }
  .note { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
          padding: 15px 17px; box-shadow: var(--shadow); }
  .note h3 { margin: 0 0 7px; font-size: 13px; letter-spacing: .03em; }
  .note p, .note li { margin: 4px 0; color: var(--muted); font-size: 13px; }
  .note ul { margin: 4px 0 0; padding-left: 18px; }
  code { background: var(--track); border-radius: 5px; padding: 1.5px 6px; font-size: 12.5px; }
  code.cmd { display: inline-block; padding: 5px 9px; font-size: 12px;
             background: color-mix(in srgb, var(--accent-soft) 60%, transparent);
             color: var(--fg); }
  .fine { margin-top: 26px; font-size: 12.5px; color: var(--muted); text-align: center; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">cmr-recon-arena · benchmark</div>
  <h1>MRI Reconstruction Leaderboard</h1>
  <p class="lede">Reconstruction quality of the models shipped with this repository,
     recomputed per sample against the fully-sampled ground truth with
     <code>scripts/calc_metric.py</code>. Pick a test set, switch metrics, and click an
     acceleration column to sort by it.</p>
  <div class="chips">
    <span class="chip">generated <b>__GENERATED__</b></span>
    <span class="chip"><b>__NGROUPS__</b> test sets</span>
    <span class="chip"><b>__NMETHODS__</b> methods</span>
    <span class="chip"><b>__NSPLITS__</b> splits scored</span>
  </div>

  <div class="controls">
    <div>
      <div class="ctl-label">Test set</div>
      <div class="tabs" id="tabs"></div>
    </div>
    <div>
      <div class="ctl-label">Metric</div>
      <div class="pills" id="metrics"></div>
    </div>
  </div>

  <div class="card"><div id="tbl"></div></div>

  <div class="notes">
    <div class="note">
      <h3>Metrics</h3>
      <ul>
        <li><b>PSNR</b> (dB) &mdash; higher is better</li>
        <li><b>SSIM</b> (%) &mdash; higher is better</li>
        <li><b>HFEN</b> (&times;1e-3) &mdash; <em>lower</em> is better</li>
      </ul>
      <p>Cells show mean &plusmn; std. Bars are min&ndash;max scaled within the
         current column and always run <em>longer&nbsp;=&nbsp;better</em> for the
         selected metric. <b>rank*</b> is the competition rank by the
         across-acceleration mean; <b>#</b> is the position in the current sort.</p>
    </div>
    <div class="note">
      <h3>Ground truth</h3>
      <p>OCMR test splits are scored against their own fully-sampled files.
         CMRxRecon prospective test splits carry no reference image and are
         scored against <code>cmrxrecon23_test_fs</code> (file names align
         because the pipeline drops the acquisition path segment).</p>
    </div>
    <div class="note">
      <h3>Reproduce</h3>
      <p>Recompute the metrics:</p>
      <p><code class="cmd">python scripts/calc_metric.py \</code><br>
         <code class="cmd">&nbsp;&nbsp;--recon-dir &lt;dir&gt; --gt-root &lt;gt&gt;</code></p>
      <p>Rebuild this page:</p>
      <p><code class="cmd">python scripts/build_leaderboard.py</code></p>
      <p>A static snapshot of the CSVs under <code>reconstructions/</code>.</p>
    </div>
  </div>
  <div class="fine">Weights for every entry live under <code>weights/&lt;test set&gt;/&lt;method&gt;.ckpt</code> ·
     rankings are computed on the fly in your browser, nothing is fetched.</div>
</div>

<script>
const DATA = __DATA__;
const state = { g: 0, metric: "PSNR", byAcc: null, flip: false };

function ranks(items, higherBetter) {          // competition rank, best = 1
  const sorted = items.slice().sort((a, b) => higherBetter ? b.v - a.v : a.v - b.v);
  const r = {}; let rank = 0, prev = null;
  sorted.forEach((it, i) => { if (it.v !== prev) { rank = i + 1; prev = it.v; } r[it.m] = rank; });
  return r;
}

function render() {
  const g = DATA.groups[state.g], M = DATA.metrics[state.metric];
  const accs = g.accs, methods = Object.keys(g.methods);
  const key = a => `${state.metric}@${a}`;

  // overall rank: metric averaged over this group's accelerations
  const avg = m => accs.reduce((s, a) => s + (g.methods[m][key(a)] || [NaN])[0], 0) / accs.length;
  const rk = ranks(methods.map(m => ({ m, v: avg(m) })), M.higher_better);

  // sort key: the clicked acceleration column, or the average across them.
  // default order is best-first; flipping the same column shows worst-first.
  const keyVal = m => state.byAcc != null
    ? ((g.methods[m][key(state.byAcc)] || [NaN])[0])
    : avg(m);
  const order = methods.slice().sort((a, b) => {
    const x = keyVal(a), y = keyVal(b);
    if (x === y) return rk[a] - rk[b];
    // desired numeric order: by default the better value comes first, which
    // is a descending sort for PSNR/SSIM and an ascending one for HFEN
    const asc = state.flip ? M.higher_better : !M.higher_better;
    return asc ? x - y : y - x;
  });

  // per-column min/max for the bars; group-wide range for the sparkline
  const colRange = {};
  accs.forEach(a => {
    const vs = methods.map(m => g.methods[m][key(a)]).filter(Boolean).map(c => c[0]);
    colRange[a] = [Math.min(...vs), Math.max(...vs)];
  });
  const allV = accs.flatMap(a => methods.map(m => (g.methods[m][key(a)] || [NaN])[0]));
  const gLo = Math.min(...allV), gHi = Math.max(...allV);
  const arrow = a => state.byAcc === a
    ? ((state.flip ? !M.higher_better : M.higher_better) ? '&#9660;' : '&#9650;')
    : '';

  // sparkline across accelerations, normalised over the whole group
  const spark = m => {
    const vs = accs.map(a => (g.methods[m][key(a)] || [NaN])[0]);
    const W = 104, H = 30, P = 5;
    const x = i => P + i * (W - 2 * P) / Math.max(vs.length - 1, 1);
    const y = v => H - P - (v - gLo) / ((gHi - gLo) || 1) * (H - 2 * P);
    const pts = vs.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
    const last = vs.length - 1;
    // neutral line so the spark matches the palette; the accent endpoint marks
    // where the method ends up at the highest acceleration
    return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" aria-hidden="true">
      <polyline points="${pts}" fill="none" stroke="var(--muted)" stroke-width="1.6"
                stroke-linejoin="round" stroke-linecap="round" opacity=".75"/>
      <circle cx="${x(last).toFixed(1)}" cy="${y(vs[last]).toFixed(1)}" r="3" fill="var(--accent)"/>
    </svg>`;
  };

  let html = `<table><thead><tr>
    <th class="pos">#</th><th class="mcol">method</th>` +
    accs.map(a => `<th class="sortable${state.byAcc === a ? ' act' : ''}" data-acc="${a}">${state.metric}@${a}X
      <span class="arr">${arrow(a)}</span></th>`).join('') +
    `<th class="mcol" title="metric across accelerations (min-max scaled)">trend</th>
     <th class="sumcol" title="mean of this metric over the group's accelerations">mean</th>
     <th class="sumcol" title="competition rank by the across-acceleration mean">rank*</th>
     </tr></thead><tbody>`;

  order.forEach((m, i) => {
    const cells = accs.map(a => {
      const c = g.methods[m][key(a)];
      if (!c) return '<td class="cell">&ndash;</td>';
      const [lo, hi] = colRange[a];
      // bar length encodes QUALITY, not raw magnitude: for HFEN (lower is
      // better) the smallest value must render as the longest bar. Worst
      // entry keeps a 15% floor so a 2-method group never shows a hairline.
      const frac = hi > lo
        ? (M.higher_better ? (c[0] - lo) : (hi - c[0])) / (hi - lo)
        : 1;
      const w = 15 + 85 * frac;
      const isBest = c[0] === (M.higher_better ? hi : lo);
      const sortedCol = state.byAcc === a;
      return `<td class="cell${isBest ? ' best' : ''}${sortedCol ? ' sorted' : ''}">
        <div class="v">${c[0].toFixed(2)}</div>
        <div class="s">&plusmn; ${c[1].toFixed(2)}</div>
        <div class="track"><i class="bar" style="width:${Math.max(w, 2).toFixed(1)}%"></i></div>
      </td>`;
    }).join('');
    const medal = i < 3 ? `<span class="medal g${i + 1}">${i + 1}</span>`
                        : `<span class="medal">${i + 1}</span>`;
    html += `<tr>
      <td class="pos">${medal}</td>
      <td class="mcol"><span class="mname">${m}</span>` +
        `<span class="mck${g.ckpt[m] ? '' : ' no'}">${g.ckpt[m] ? 'ckpt' : 'no ckpt'}</span></td>
      ${cells}
      <td class="spark">${spark(m)}</td>
      <td class="sumcol numcol">${avg(m).toFixed(2)}</td>
      <td class="sumcol rankcol">${rk[m]}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('tbl').innerHTML = html;
  document.querySelectorAll('thead th[data-acc]').forEach(th => th.onclick = () => {
    const a = th.dataset.acc;
    if (state.byAcc === a) state.flip = !state.flip;
    else { state.byAcc = a; state.flip = false; }   // first click on a column = best-first
    render();
  });
}

document.getElementById('tabs').innerHTML = DATA.groups.map((g, i) =>
  `<button class="tab${i === state.g ? ' on' : ''}" data-i="${i}">
     <span class="t-name">${g.name}</span>
     <span class="t-meta">${Object.keys(g.methods).length} methods` +
     `${g.n ? ` &middot; n=${g.n}` : ''} &middot; ${g.accs.map(a => a + 'X').join('/')}</span>
   </button>`).join('');
document.getElementById('metrics').innerHTML = Object.keys(DATA.metrics).map(k =>
  `<button class="pill${k === state.metric ? ' on' : ''}" data-m="${k}">` +
  `${DATA.metrics[k].label} ${DATA.metrics[k].higher_better ? '&#8593;' : '&#8595;'}</button>`).join('');
document.getElementById('tabs').onclick = e => {
  const b = e.target.closest('button'); if (!b) return;
  state.g = +b.dataset.i; state.byAcc = null;
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('on', x === b));
  render();
};
document.getElementById('metrics').onclick = e => {
  const b = e.target.closest('button'); if (!b) return;
  state.metric = b.dataset.m;
  document.querySelectorAll('.pill').forEach(x => x.classList.toggle('on', x === b));
  render();
};
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    data = build()
    payload = json.loads(data)
    print(f"wrote {OUT.relative_to(ROOT)}  "
          f"({len(payload['groups'])} groups, "
          f"{len({m for g in payload['groups'] for m in g['methods']})} methods, "
          f"{len(data) // 1024} KB of embedded data)")
