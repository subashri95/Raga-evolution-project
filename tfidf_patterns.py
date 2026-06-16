#!/usr/bin/env python3
"""
TF-IDF analysis of melodic patterns across Bhairavi groups.

Documents = groups (Group1, Group2, Group3)
Terms     = pattern labels from annotations
Corpus    = all three groups (the raga as a whole)
"""
from __future__ import annotations

import csv
import math
import os
from collections import defaultdict

import io, base64

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

from data_utils import build_stem_map, build_group_counts, load_annotations

BASE             = os.path.dirname(os.path.abspath(__file__))
ANNOTATIONS_PATH = os.path.join(BASE, "Annotations", "annotations-4.csv")
REPORT_PATH      = os.path.join(BASE, "tfidf_report.html")

PALETTE = {
    "Group1": "#7F2020",
    "Group2": "#4a7c59",
    "Group3": "#3a5a8c",
}

def load_annotations(path: str) -> list[tuple[str, str]]:
    """Return list of (track_stem, pattern_label)."""
    records: list[tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            track = row.get("track", "").strip()
            label = row.get("label", "").strip()
            if track and label:
                records.append((track, label))
    return records


# ---------------------------------------------------------------------------
# Build group-pattern counts
# ---------------------------------------------------------------------------

stem_map  = build_stem_map(
    os.path.join(BASE, "manifest3.csv"),  # canonical
    os.path.join(BASE, "manifest.csv"),   # fallback
)
annotations = load_annotations(ANNOTATIONS_PATH)
group_pattern_counts, unmatched_tracks = build_group_counts(annotations, stem_map)

groups  = sorted(group_pattern_counts.keys())
all_patterns = sorted({p for counts in group_pattern_counts.values() for p in counts})

print(f"Groups: {groups}")
print(f"Unique patterns: {len(all_patterns)}")
print(f"Annotations matched: {sum(sum(c.values()) for c in group_pattern_counts.values())}")
if unmatched_tracks:
    print(f"Unmatched tracks ({len(unmatched_tracks)}): {unmatched_tracks[:5]} ...")


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------
# TF  = count(pattern, group) / total_patterns_in_group
# IDF = log(N / df)  where df = number of groups containing pattern
# TF-IDF = TF * IDF
# ---------------------------------------------------------------------------

N = len(groups)

# TF per group
tf: dict[str, dict[str, float]] = {}
for grp in groups:
    total = sum(group_pattern_counts[grp].values()) or 1
    tf[grp] = {p: group_pattern_counts[grp].get(p, 0) / total for p in all_patterns}

# IDF across groups
df: dict[str, int] = {
    p: sum(1 for grp in groups if group_pattern_counts[grp].get(p, 0) > 0)
    for p in all_patterns
}
idf: dict[str, float] = {
    p: math.log(N / df[p]) if df[p] > 0 else 0.0
    for p in all_patterns
}

# TF-IDF
tfidf: dict[str, dict[str, float]] = {
    grp: {p: tf[grp][p] * idf[p] for p in all_patterns}
    for grp in groups
}

# Top-N distinctive patterns per group (sorted by TF-IDF desc)
TOP_N = 15
top_per_group: dict[str, list[tuple[str, float, int]]] = {}
for grp in groups:
    ranked = sorted(
        [(p, tfidf[grp][p], group_pattern_counts[grp].get(p, 0)) for p in all_patterns],
        key=lambda x: -x[1],
    )
    top_per_group[grp] = [(p, score, cnt) for p, score, cnt in ranked if score > 0][:TOP_N]

# ---------------------------------------------------------------------------
# Enrichment score
# Enrichment(p, g) = TF(p, g) / corpus_TF(p)
# > 1  → pattern overindexes in this group
# = 1  → pattern matches corpus average
# < 1  → pattern underindexes
# Only computed for patterns with total count >= MIN_COUNT
# ---------------------------------------------------------------------------
MIN_COUNT = 2
total_all  = sum(sum(c.values()) for c in group_pattern_counts.values())
corpus_tf  = {p: sum(group_pattern_counts[g].get(p, 0) for g in groups) / total_all
              for p in all_patterns}
group_totals = {g: sum(group_pattern_counts[g].values()) or 1 for g in groups}

enrichment: dict[str, dict[str, float]] = {}
for g in groups:
    enrichment[g] = {}
    for p in all_patterns:
        total_p = sum(group_pattern_counts[g2].get(p, 0) for g2 in groups)
        if total_p < MIN_COUNT:
            continue
        ctf = corpus_tf[p]
        enrichment[g][p] = (tf[g][p] / ctf) if ctf > 0 else 0.0

# Top overindexing patterns per group (enrichment > 1, sorted desc)
TOP_ENRICH = 20
top_enriched: dict[str, list[tuple[str, float, int, int]]] = {}
for g in groups:
    ranked = sorted(
        [(p, enrichment[g][p], group_pattern_counts[g].get(p, 0),
          sum(group_pattern_counts[g2].get(p, 0) for g2 in groups))
         for p in enrichment[g] if enrichment[g][p] > 1],
        key=lambda x: -x[1],
    )
    top_enriched[g] = ranked[:TOP_ENRICH]

# ---------------------------------------------------------------------------
# HTML report — interactive table, no plots
# ---------------------------------------------------------------------------
import json as _json

n_total = sum(sum(c.values()) for c in group_pattern_counts.values())

# Serialise pattern data for JS
rows_json = _json.dumps([
    {
        "pattern": p,
        "total":   sum(group_pattern_counts[g].get(p, 0) for g in groups),
        "df":      df[p],
        "groups":  {
            g: {
                "count": group_pattern_counts[g].get(p, 0),
                "enr":   round(enrichment[g].get(p, -1), 4),  # -1 = below min_count
            }
            for g in groups
        },
    }
    for p in all_patterns
], ensure_ascii=False)

grp_meta_json = _json.dumps({
    g: {
        "color": PALETTE.get(g, "#888"),
        "total": sum(group_pattern_counts[g].values()),
    }
    for g in groups
}, ensure_ascii=False)

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pattern Analysis — Bhairavi Raga</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; background: #f7f5f0;
       color: #2c2c2c; line-height: 1.6; }

/* ---- page sections ---- */
.page-header { background: #0f0f14; color: #f0ede8; padding: 1.5rem 2.5rem; }
.page-header h1 { font-weight: 300; font-size: 1.6rem; letter-spacing: -.02em; }
.page-header p  { font-size: .82rem; color: #999; margin-top: .25rem; }

.description { background: #fff; border-bottom: 1px solid #e0ddd8;
               padding: 1.2rem 2.5rem; }
.description h2 { font-size: .75rem; font-weight: 700; letter-spacing: .12em;
                  text-transform: uppercase; color: #888; margin-bottom: .7rem; }
.description p  { font-size: .86rem; color: #333; max-width: 860px;
                  line-height: 1.75; margin-bottom: .5rem; }
.formula { font-family: 'Courier New', monospace; font-size: .9rem;
           background: #f7f5f0; display: inline-block; padding: .45rem 1rem;
           border-radius: 6px; border-left: 3px solid #6baed6;
           margin: .4rem 0 .5rem; color: #1a1a1a; }

/* ---- sticky toolbar ---- */
.toolbar { background: #fff; border-bottom: 1px solid #e0ddd8;
           padding: .75rem 2.5rem; display: flex; flex-wrap: wrap;
           gap: .7rem; align-items: center;
           position: sticky; top: 0; z-index: 20; }
.toolbar label { font-size: .78rem; color: #555;
                 display: flex; align-items: center; gap: .4rem; }
input[type=text]   { font-size: .82rem; padding: .3rem .6rem; border: 1px solid #ddd;
                     border-radius: 6px; width: 190px; outline: none; }
input[type=number] { font-size: .82rem; padding: .3rem .5rem; border: 1px solid #ddd;
                     border-radius: 6px; width: 65px; outline: none; }
input:focus { border-color: #aaa; }
.sep { width: 1px; height: 22px; background: #e0ddd8; flex-shrink: 0; }
.lbl { font-size: .75rem; color: #888; }
.btn { font-size: .74rem; padding: .28rem .65rem; border-radius: 999px;
       border: 1px solid #ddd; background: #fff; cursor: pointer;
       white-space: nowrap; transition: all .15s; }
.btn:hover  { background: #0f0f14; color: #fff; border-color: #0f0f14; }
.btn.active { background: #0f0f14; color: #fff; border-color: #0f0f14; }
#row-count  { font-size: .76rem; color: #aaa; margin-left: auto; }

/* ---- table wrapper ---- */
.table-wrap { padding: 1.5rem 2.5rem 3rem; overflow-x: auto; }

/* ---- table ---- */
table { border-collapse: collapse; width: 100%; font-size: .82rem;
        background: #fff; border-radius: 8px; overflow: hidden;
        box-shadow: 0 1px 8px rgba(0,0,0,.07); }

/* single header row — no sticky, no stacking issues */
thead th { background: #1a1a2e; color: #c8c8d8; font-weight: 600;
           font-size: .7rem; letter-spacing: .07em; text-transform: uppercase;
           padding: .6rem .75rem; text-align: center;
           cursor: pointer; user-select: none; white-space: nowrap;
           border-right: 1px solid #2a2a3e; }
thead th:first-child { text-align: left; }
thead th:last-child  { border-right: none; }
thead th:hover { background: #2a2a3e; color: #fff; }
thead th.sorted-asc::after  { content: ' ▲'; font-size: .62rem; opacity: .7; }
thead th.sorted-desc::after { content: ' ▼'; font-size: .62rem; opacity: .7; }

/* group colour band under header */
thead th.g0 { border-top: 3px solid """ + PALETTE.get(groups[0] if groups else 'G1', '#888') + """; }
thead th.g1 { border-top: 3px solid """ + PALETTE.get(groups[1] if len(groups)>1 else 'G2', '#888') + """; }
thead th.g2 { border-top: 3px solid """ + PALETTE.get(groups[2] if len(groups)>2 else 'G3', '#888') + """; }

tbody tr { border-bottom: 1px solid #f0ede8; transition: background .08s; }
tbody tr:last-child { border-bottom: none; }
tbody tr:hover { background: #f0ede8; }
td { padding: .4rem .75rem; text-align: right; vertical-align: middle; }
td.pat { text-align: left; font-family: 'Courier New', monospace; font-size: .84rem;
         font-weight: 500; color: #1a1a1a; white-space: nowrap; }
td.zero { color: #ccc; }
td.num  { font-variant-numeric: tabular-nums; }

.bar-cell { display: inline-flex; align-items: center; gap: .4rem; }
.bar      { height: 7px; border-radius: 999px; flex-shrink: 0; }

.prev-cell { font-family: 'Courier New', monospace; font-size: .78rem; }
.prev-high { font-weight: 700; color: #1b5e20; }
.prev-mid  { color: #388e3c; }
.prev-low  { color: #999; }
.prev-none { color: #ccc; }
.prev-best { background: #e8f5e9; }

.df-badge { display: inline-block; width: 20px; height: 20px; border-radius: 50%;
            line-height: 20px; font-size: .68rem; font-weight: 700;
            text-align: center; color: #fff; }
.badge    { display: inline-block; padding: .15rem .55rem; border-radius: 4px;
            font-size: .75rem; font-weight: 600; color: #fff; margin: .1rem; }
</style>
</head>
<body>

<div class="page-header">
  <h1>Bhairavi Raga — Pattern Analysis</h1>
  <p id="header-meta"></p>
</div>

<div class="description">
  <h2>Prevalence Score</h2>
  <p>
    For each pattern <em>p</em> and group <em>g</em>, the <strong>prevalence score</strong>
    measures how much more (or less) often the pattern appears in that group compared to
    what would be expected if it were distributed in proportion to each group's size:
  </p>
  <p class="formula">
    Prevalence(p, g) &nbsp;=&nbsp; TF(p, g) &nbsp;/&nbsp; TF<sub>corpus</sub>(p)
  </p>
  <p>
    <strong>TF(p, g)</strong> = count(p, g) / total annotations in <em>g</em> &nbsp;&nbsp;·&nbsp;&nbsp;
    <strong>TF<sub>corpus</sub>(p)</strong> = total count(p) / total annotations across all groups.
    <br>
    <strong>1×</strong> = appears exactly as expected &nbsp;·&nbsp;
    <strong>&gt;1×</strong> = overindexes (e.g. 3× = three times the corpus average) &nbsp;·&nbsp;
    <strong>&lt;1×</strong> = underrepresented.
    Prevalence is only shown for patterns with &ge;""" + str(MIN_COUNT) + """ total occurrences.
    The <span style="background:#e8f5e9;padding:0 4px;border-radius:3px">green</span> cell
    in each row marks the group where the pattern is most prevalent.
  </p>
</div>

<div class="toolbar">
  <label>Search &nbsp;<input type="text" id="search" placeholder="pattern name…"></label>
  <div class="sep"></div>
  <label>Min total occurrences &nbsp;<input type="number" id="min-count" value="1" min="1"></label>
  <div class="sep"></div>
  <span class="lbl">Show:</span>
  <button class="btn active" data-df="0">All</button>
  <button class="btn" data-df="1">Exclusive (1 group)</button>
  <button class="btn" data-df="2">2 groups</button>
  <button class="btn" data-df="3">All 3 groups</button>
  <div class="sep"></div>
  <span class="lbl">Sort by prevalence in:</span>
  <span id="prev-btns"></span>
  <span id="row-count"></span>
</div>

<div class="table-wrap">
  <table>
    <thead id="thead"></thead>
    <tbody id="tbody"></tbody>
  </table>
</div>

<script>
const ROWS     = """ + rows_json + """;
const GRP_META = """ + grp_meta_json + """;
const GROUPS   = Object.keys(GRP_META);
const N_TOTAL  = """ + str(n_total) + """;

// header meta
document.getElementById('header-meta').innerHTML =
  GROUPS.map(g =>
    `<span class="badge" style="background:${GRP_META[g].color}">${g} — ${GRP_META[g].total} occurrences</span>`
  ).join(' ') + ` &nbsp;·&nbsp; ${ROWS.length} unique patterns &nbsp;·&nbsp; ${N_TOTAL} total occurrences`;

// prevalence sort buttons
const prevBtnsEl = document.getElementById('prev-btns');
GROUPS.forEach(g => {
  const b = document.createElement('button');
  b.className = 'btn';
  b.textContent = g;
  b.style.cssText = `border-color:${GRP_META[g].color};color:${GRP_META[g].color}`;
  b.dataset.prevGroup = g;
  b.addEventListener('click', () => {
    sortKey = 'enr_' + g; sortDir = -1;
    prevBtnsEl.querySelectorAll('button').forEach(x => {
      x.classList.remove('active');
      x.style.color = GRP_META[x.dataset.prevGroup].color;
    });
    b.classList.add('active');
    b.style.color = '#fff';
    render();
  });
  prevBtnsEl.appendChild(b);
});

// build single-row thead
const colDefs = [
  { key: 'pattern', label: 'Pattern' },
  { key: 'total',   label: 'Total' },
  ...GROUPS.flatMap((g, gi) => [
    { key: `count_${g}`, label: `${g} count`, gi },
    { key: `enr_${g}`,   label: `${g} prevalence`, gi },
  ]),
  { key: 'df', label: 'Groups' },
];

document.getElementById('thead').innerHTML =
  '<tr>' + colDefs.map(c => {
    const cls = c.gi !== undefined ? ` class="g${c.gi}"` : '';
    return `<th data-key="${c.key}"${cls}>${c.label}</th>`;
  }).join('') + '</tr>';

// sort state
let sortKey = 'total', sortDir = -1, dfFilter = 0;

document.getElementById('thead').querySelectorAll('th').forEach(th => {
  th.addEventListener('click', () => {
    if (sortKey === th.dataset.key) sortDir *= -1;
    else { sortKey = th.dataset.key; sortDir = -1; }
    render();
  });
});

document.querySelectorAll('[data-df]').forEach(btn => {
  btn.addEventListener('click', () => {
    dfFilter = +btn.dataset.df;
    document.querySelectorAll('[data-df]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    render();
  });
});

document.getElementById('search').addEventListener('input', render);
document.getElementById('min-count').addEventListener('input', render);

const maxCount = {};
GROUPS.forEach(g => maxCount[g] = Math.max(1, ...ROWS.map(r => r.groups[g].count)));

function render() {
  const q      = document.getElementById('search').value.trim().toLowerCase();
  const minCnt = parseInt(document.getElementById('min-count').value) || 1;

  let data = ROWS.filter(r =>
    r.total >= minCnt &&
    (dfFilter === 0 || r.df === dfFilter) &&
    (!q || r.pattern.toLowerCase().includes(q))
  );

  data.sort((a, b) => {
    let va, vb;
    if (sortKey === 'pattern') return sortDir * a.pattern.localeCompare(b.pattern);
    if (sortKey === 'total')   { va = a.total; vb = b.total; }
    else if (sortKey === 'df') { va = a.df;    vb = b.df; }
    else if (sortKey.startsWith('count_')) {
      const g = sortKey.slice(6); va = a.groups[g].count; vb = b.groups[g].count;
    } else {
      const g = sortKey.slice(4); va = a.groups[g].enr; vb = b.groups[g].enr;
    }
    return sortDir * (va - vb);
  });

  document.getElementById('thead').querySelectorAll('th').forEach(th => {
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (th.dataset.key === sortKey)
      th.classList.add(sortDir === 1 ? 'sorted-asc' : 'sorted-desc');
  });

  const bestG = r => GROUPS.reduce((best, g) =>
    (r.groups[g].enr > r.groups[best].enr) ? g : best, GROUPS[0]);

  document.getElementById('tbody').innerHTML = data.map(r => {
    const bg = bestG(r);
    const groupCells = GROUPS.map(g => {
      const { count, enr } = r.groups[g];
      const color = GRP_META[g].color;
      const barW  = Math.round(count / maxCount[g] * 52);

      const cntTd = count === 0
        ? `<td class="zero">—</td>`
        : `<td class="num"><div class="bar-cell">
             <div class="bar" style="width:${barW}px;background:${color};opacity:.65"></div>
             ${count}
           </div></td>`;

      let prevTd;
      if (enr < 0) {
        prevTd = `<td class="prev-none">—</td>`;
      } else {
        const isTop = g === bg && enr >= 1;
        const cls   = enr >= 2 ? 'prev-high' : enr >= 1.1 ? 'prev-mid' : 'prev-low';
        prevTd = `<td class="prev-cell ${cls}${isTop ? ' prev-best' : ''}">${enr.toFixed(2)}&times;</td>`;
      }
      return cntTd + prevTd;
    }).join('');

    const dfColor = r.df === GROUPS.length ? '#555' : GRP_META[bg].color;

    return `<tr>
      <td class="pat">${r.pattern}</td>
      <td class="num">${r.total}</td>
      ${groupCells}
      <td><span class="df-badge" style="background:${dfColor}">${r.df}</span></td>
    </tr>`;
  }).join('');

  document.getElementById('row-count').textContent = `${data.length} / ${ROWS.length} patterns`;
}

render();
</script>
</body>
</html>
"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nReport written to: {REPORT_PATH}")
print(str(len(all_patterns)) + " patterns · " + str(n_total) + " occurrences")
