#!/usr/bin/env python3
from __future__ import annotations
"""
Carnatic / Hindustānī intonation analysis pipeline.

Commands
--------
  extract   Extract and cache pitch tracks (tonic sidecar must exist).
  run       Full pipeline: process manifest CSV, compare all groups, write HTML report.

Manifest CSV format (header row required):
    path,group,tonic
    /recordings/file1.wav,GroupA,293.66
    /recordings/file2.wav,GroupB,220.00
    ...

The 'tonic' column is optional per-row; if absent or empty the pipeline falls
back to a _tonic.txt sidecar file next to the audio.

With N groups the report runs:
  - Kruskal-Wallis omnibus test per (svara, metric)
  - Pairwise Mann-Whitney U post-hoc for all N*(N-1)/2 pairs
  - Single Bonferroni correction denominator across all pairwise tests
"""

import base64
import csv
import io
import logging
import os
import time
import warnings
from collections import defaultdict
from itertools import combinations

warnings.filterwarnings("ignore")

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, kruskal, kurtosis, mannwhitneyu, skew
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("subashri")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger("subashri").setLevel(level)
    logging.getLogger("subashri").addHandler(handler)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SVARA_GRID = np.array(
    [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],
    dtype=np.float64,
)
SVARA_NAMES = [
    "S", "R1", "R2/G1", "R3/G2", "G3", "M1",
    "M2", "P", "D1", "D2/N1", "N2/D3", "N3",
]

WINDOW_CENTS = 50
KDE_BW       = 0.03   # reduced from 0.075 for sharper svara peaks
KDE_N_POINTS = 1200

PRESENCE_FACTOR    = 1.5
PRESENCE_THRESHOLD = PRESENCE_FACTOR / 1200.0

METRICS = [
    "peak_loc", "offset_cents", "peak_height",
    "peak_width", "skewness", "kurtosis",
]
METRIC_LABELS = {
    "peak_loc":     "Peak location (cents from tonic)",
    "offset_cents": "Intonation offset from ET (cents)",
    "peak_height":  "Peak height (density)",
    "peak_width":   "Peak width / FWHM (cents)",
    "skewness":     "Skewness",
    "kurtosis":     "Excess kurtosis",
}

# Colour palette (cycles for > 3 groups)
_PALETTE = [
    "#7F2020",  # deep red
    "#869B7E",  # sage green
    "#C9CAAC",  # warm stone
    "#a0b9d9",  # steel blue (fallback)
    "#c7b8e0",  # pale purple (fallback)
]


def _group_colors(labels: list) -> dict:
    """Map each group label to a hex colour, deterministically by sort order."""
    return {label: _PALETTE[i % len(_PALETTE)] for i, label in enumerate(sorted(labels))}


def _apply_plot_style() -> None:
    """Apply a clean, academic, pastel matplotlib style to all subsequent plots."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#f9f9f9",
        "axes.grid":         True,
        "grid.color":        "#e0e0e0",
        "grid.linewidth":    0.6,
        "grid.alpha":        1.0,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    "#bbbbbb",
        "axes.linewidth":    0.8,
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.titleweight":  "normal",
        "axes.labelsize":    10,
        "axes.labelcolor":   "#333333",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "xtick.color":       "#555555",
        "ytick.color":       "#555555",
        "legend.frameon":    True,
        "legend.framealpha": 0.92,
        "legend.edgecolor":  "#dddddd",
        "legend.fontsize":   9,
        "legend.fancybox":   True,
        "lines.linewidth":   1.8,
        "patch.linewidth":   0.4,
    })


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

# Note name → frequency in octave 3 (A3 = 220 Hz, C3 = 130.81 Hz)
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTE_ALIASES = {"Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
                 "Ab": "G#", "Bb": "A#", "Cb": "B"}

def note_to_hz(note: str) -> float:
    """Convert a note name (e.g. 'A', 'F#', 'Bb') to Hz in octave 3."""
    note = note.strip()
    note = _NOTE_ALIASES.get(note, note)
    if note not in _NOTE_NAMES:
        raise ValueError(
            f"Unrecognised note name '{note}'. "
            f"Valid names: {_NOTE_NAMES + list(_NOTE_ALIASES.keys())}"
        )
    semitone = _NOTE_NAMES.index(note)
    # C3 = MIDI 48; A4 = 440 Hz = MIDI 69
    midi = 48 + semitone
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def parse_tonic(value: str | float) -> float:
    """Accept either a Hz float or a note-name string; return Hz float."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except ValueError:
        return note_to_hz(value)


def hz_to_cents(freqs: np.ndarray, tonic: float) -> np.ndarray:
    return 1200.0 * np.log2(np.asarray(freqs, dtype=np.float64) / tonic)


def compute_fwhm(x: np.ndarray, y: np.ndarray, peak_idx: int) -> float:
    half = y[peak_idx] / 2.0
    left = peak_idx
    while left > 0 and y[left] > half:
        left -= 1
    right = peak_idx
    while right < len(y) - 1 and y[right] > half:
        right += 1
    return float(x[right] - x[left])


def compute_kde(cents_values: np.ndarray, bw_method: float = KDE_BW):
    kde = gaussian_kde(cents_values, bw_method=bw_method)
    x = np.linspace(0.0, 1200.0, KDE_N_POINTS)
    y = kde(x)
    y /= np.trapz(y, x)
    return x, y


def analyse_file(audio_path: str, tonic: float | None = None) -> tuple:
    """
    Load (or extract+cache) pitch, compute density-normalised KDE, and return
    per-svara statistics.

    Parameters
    ----------
    audio_path : path to audio file
    tonic      : Hz — if None, falls back to _tonic.txt sidecar

    Returns
    -------
    x           : ndarray (KDE_N_POINTS,)
    y           : ndarray (KDE_N_POINTS,) — density-normalised KDE
    svara_stats : dict[name -> dict | None]
    cents       : ndarray — raw tonic-normalised pitch values (for histogram overlays)
    """
    pitch_track = _load_or_extract_pitch(audio_path)
    tonic = _load_tonic(audio_path, tonic)

    pitch_vals = pitch_track[:, 1]
    n_total = len(pitch_vals)
    pitch_vals = pitch_vals[pitch_vals > 0]
    n_voiced = len(pitch_vals)
    if n_voiced == 0:
        raise ValueError(f"No non-zero pitch values found in: {audio_path}")
    log.info(
        "Pitch frames: %d total, %d voiced (%.0f%%)",
        n_total, n_voiced, 100 * n_voiced / n_total,
    )

    cents = hz_to_cents(pitch_vals, tonic) % 1200.0
    log.debug("Computing KDE (bw=%.3f, %d points)", KDE_BW, KDE_N_POINTS)
    x, y = compute_kde(cents)

    svara_stats: dict = {}
    for svara, name in zip(SVARA_GRID, SVARA_NAMES):
        kde_mask = (x >= svara - WINDOW_CENTS) & (x <= svara + WINDOW_CENTS)
        if not np.any(kde_mask):
            svara_stats[name] = None
            continue

        local_x = x[kde_mask]
        local_y = y[kde_mask]

        # Require a genuine local maximum — reject shoulders of adjacent peaks
        peak_idxs, _ = find_peaks(local_y)
        if len(peak_idxs) == 0:
            svara_stats[name] = None
            continue
        # Among genuine peaks, take the tallest
        idx_max     = peak_idxs[int(np.argmax(local_y[peak_idxs]))]
        peak_loc    = float(local_x[idx_max])
        peak_height = float(local_y[idx_max])

        global_idx = int(np.clip(np.searchsorted(x, peak_loc), 0, len(x) - 1))
        peak_width = compute_fwhm(x, y, global_idx)

        raw_mask = (cents >= svara - WINDOW_CENTS) & (cents <= svara + WINDOW_CENTS)
        raw_vals = cents[raw_mask]

        if len(raw_vals) >= 10:
            peak_skew = float(skew(raw_vals))
            peak_kurt = float(kurtosis(raw_vals, fisher=True))
        else:
            peak_skew = float("nan")
            peak_kurt = float("nan")

        present = peak_height >= PRESENCE_THRESHOLD
        svara_stats[name] = {
            "peak_loc":     peak_loc,
            "offset_cents": peak_loc - svara,
            "peak_height":  peak_height,
            "peak_width":   peak_width,
            "skewness":     peak_skew,
            "kurtosis":     peak_kurt,
            "n_raw":        int(np.sum(raw_mask)),
            "present":      present,
        }
        log.debug(
            "  %-8s  loc=%6.1f  offset=%+.1f  height=%.5f  width=%5.1f  "
            "skew=%+.2f  kurt=%+.2f  n_raw=%d  %s",
            name, peak_loc, peak_loc - svara, peak_height, peak_width,
            peak_skew if not np.isnan(peak_skew) else 0.0,
            peak_kurt if not np.isnan(peak_kurt) else 0.0,
            int(np.sum(raw_mask)),
            "PRESENT" if present else "absent",
        )

    n_present = sum(1 for s in svara_stats.values() if s is not None and s["present"])
    log.info("Svaras present: %d / %d", n_present, len(SVARA_NAMES))
    return x, y, svara_stats, cents


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_or_extract_pitch(audio_path: str) -> np.ndarray:
    cache = os.path.splitext(audio_path)[0] + ".tsv"
    if os.path.exists(cache):
        log.info("Pitch cache hit: %s", os.path.basename(cache))
        track = np.loadtxt(cache, delimiter="\t")
        log.debug("Loaded %d frames from cache", len(track))
        return track
    from compiam.melody.pitch_extraction import Melodia
    log.info("Extracting pitch: %s", audio_path)
    t0 = time.time()
    track = Melodia().extract(audio_path)
    log.info("Extraction complete: %d frames in %.1fs", len(track), time.time() - t0)
    np.savetxt(cache, track, delimiter="\t")
    log.debug("Cached pitch track to: %s", cache)
    return track


def _load_tonic(audio_path: str, tonic: str | float | None = None) -> float:
    """Resolve tonic to Hz. Accepts a Hz float, a note name string, or None (sidecar fallback)."""
    if tonic is not None:
        hz = parse_tonic(tonic)
        log.debug("Tonic from manifest: %.2f Hz", hz)
        return hz
    sidecar = os.path.splitext(audio_path)[0] + "_tonic.txt"
    if not os.path.exists(sidecar):
        raise FileNotFoundError(
            f"No tonic in manifest and no sidecar found: {sidecar}"
        )
    raw = np.loadtxt(sidecar, dtype=str).item()
    hz = parse_tonic(raw)
    log.debug("Tonic from sidecar: %.2f Hz (%s)", hz, sidecar)
    return hz


def load_manifest(manifest_path: str) -> dict:
    """
    Return {group_label: [{"path": str, "tonic": float | None}, ...]}

    CSV must have 'path' and 'group' columns. Optional 'tonic' column (Hz)
    takes precedence over a _tonic.txt sidecar.
    """
    groups: dict = defaultdict(list)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_tonic = "tonic" in (reader.fieldnames or [])
        for row in reader:
            tonic_val = None
            if has_tonic and row.get("tonic", "").strip():
                tonic_val = row["tonic"].strip()  # kept as string; parse_tonic handles Hz or note name
            groups[row["group"].strip()].append({
                "path":  row["path"].strip(),
                "tonic": tonic_val,
            })
    return dict(groups)


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def _collect_metric(peaks: list, svara: str, metric: str) -> list:
    return [
        p[svara][metric]
        for p in peaks
        if p[svara] is not None and not np.isnan(p[svara][metric])
    ]


def _nan_summary(vals: list) -> dict:
    return {
        "mean":   np.mean(vals)        if vals         else float("nan"),
        "std":    np.std(vals, ddof=1) if len(vals) > 1 else float("nan"),
        "median": np.median(vals)      if vals         else float("nan"),
        "n":      len(vals),
    }


def compare_groups(all_peaks: dict, alpha: float = 0.05) -> tuple:
    """
    Run omnibus Kruskal-Wallis and pairwise Mann-Whitney U tests across all
    groups for every (svara, metric) combination.

    Parameters
    ----------
    all_peaks : {group_label: [svara_stats_dict, ...]}
    alpha     : family-wise error rate

    Returns
    -------
    omnibus          : dict[svara][metric]      -> omnibus stats dict
    pairwise         : dict[(la,lb)][svara][metric] -> pairwise stats dict
    n_peaks_omnibus  : dict  (KW on n_present svaras per recording)
    n_peaks_pairwise : dict[(la,lb)] -> MWU stats dict
    n_tests_omnibus  : int
    n_tests_pairwise : int
    """
    labels = sorted(all_peaks.keys())
    pairs  = list(combinations(labels, 2))

    n_sv_mt          = len(SVARA_NAMES) * len(METRICS)
    n_tests_omnibus  = n_sv_mt
    n_tests_pairwise = len(pairs) * n_sv_mt  # single denominator across all pairs

    log.info(
        "Statistical comparison: %d groups, %d pair(s), "
        "%d omnibus tests, %d pairwise tests",
        len(labels), len(pairs), n_tests_omnibus, n_tests_pairwise,
    )

    # ---- Omnibus (Kruskal-Wallis) ----------------------------------------
    omnibus: dict = {}
    for name in SVARA_NAMES:
        omnibus[name] = {}
        for metric in METRICS:
            groups_data = {
                label: _collect_metric(all_peaks[label], name, metric)
                for label in labels
            }
            summaries = {label: _nan_summary(vals) for label, vals in groups_data.items()}
            entry = {
                "summaries":    summaries,
                "H":            float("nan"),
                "p_raw":        float("nan"),
                "p_bonferroni": float("nan"),
                "significant":  False,
            }
            valid = [v for v in groups_data.values() if len(v) >= 1]
            if len(valid) >= 2:
                try:
                    H, p = kruskal(*valid)
                    p_bonf = min(p * n_tests_omnibus, 1.0)
                    entry.update({
                        "H":            float(H),
                        "p_raw":        float(p),
                        "p_bonferroni": p_bonf,
                        "significant":  p_bonf < alpha,
                    })
                except ValueError:
                    pass  # all values identical — test undefined
            omnibus[name][metric] = entry

    # ---- Pairwise (Mann-Whitney U) ----------------------------------------
    pairwise: dict = {}
    for la, lb in pairs:
        pairwise[(la, lb)] = {}
        for name in SVARA_NAMES:
            pairwise[(la, lb)][name] = {}
            for metric in METRICS:
                vals_a = _collect_metric(all_peaks[la], name, metric)
                vals_b = _collect_metric(all_peaks[lb], name, metric)
                entry = {
                    "summary_a":    _nan_summary(vals_a),
                    "summary_b":    _nan_summary(vals_b),
                    "p_raw":        float("nan"),
                    "p_bonferroni": float("nan"),
                    "significant":  False,
                    "effect_r":     float("nan"),
                }
                if len(vals_a) >= 1 and len(vals_b) >= 1:
                    try:
                        stat, p = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                        r      = 1.0 - (2.0 * stat) / (len(vals_a) * len(vals_b))
                        p_bonf = min(p * n_tests_pairwise, 1.0)
                        entry.update({
                            "p_raw":        float(p),
                            "p_bonferroni": p_bonf,
                            "significant":  p_bonf < alpha,
                            "effect_r":     float(r),
                        })
                    except ValueError:
                        pass  # all values identical — test undefined
                pairwise[(la, lb)][name][metric] = entry

    # ---- n_peaks (number of present svaras per recording) -----------------
    n_peaks_by_group = {
        label: [
            sum(1 for name in SVARA_NAMES if p[name] is not None and p[name]["present"])
            for p in peaks_list
        ]
        for label, peaks_list in all_peaks.items()
    }

    n_peaks_omnibus: dict = {
        "summaries":    {l: _nan_summary(v) for l, v in n_peaks_by_group.items()},
        "H":            float("nan"),
        "p_raw":        float("nan"),
        "p_bonferroni": float("nan"),
        "significant":  False,
    }
    valid_np = [v for v in n_peaks_by_group.values() if len(v) >= 1]
    if len(valid_np) >= 2:
        try:
            H, p = kruskal(*valid_np)
            p_bonf = min(p * (n_tests_omnibus + 1), 1.0)
            n_peaks_omnibus.update({
                "H": float(H), "p_raw": float(p),
                "p_bonferroni": p_bonf, "significant": p_bonf < alpha,
            })
        except ValueError:
            pass  # all values identical — test undefined

    n_peaks_pairwise: dict = {}
    for la, lb in pairs:
        va, vb = n_peaks_by_group[la], n_peaks_by_group[lb]
        entry = {
            "summary_a": _nan_summary(va), "summary_b": _nan_summary(vb),
            "p_raw": float("nan"), "p_bonferroni": float("nan"),
            "significant": False, "effect_r": float("nan"),
        }
        if len(va) >= 1 and len(vb) >= 1:
            try:
                stat, p = mannwhitneyu(va, vb, alternative="two-sided")
                r = 1.0 - (2.0 * stat) / (len(va) * len(vb))
                p_bonf = min(p * (n_tests_pairwise + len(pairs)), 1.0)
                entry.update({
                    "p_raw": float(p), "p_bonferroni": p_bonf,
                    "significant": p_bonf < alpha, "effect_r": float(r),
                })
            except ValueError:
                pass  # all values identical — test undefined
        n_peaks_pairwise[(la, lb)] = entry

    return omnibus, pairwise, n_peaks_omnibus, n_peaks_pairwise, n_tests_omnibus, n_tests_pairwise


# ---------------------------------------------------------------------------
# Qualitative interpretation
# ---------------------------------------------------------------------------

def _interpret(svara: str, metric: str, mean_a: float, mean_b: float,
               effect_r: float, label_a: str, label_b: str) -> str:
    """Return a human-readable musicological interpretation of a pairwise difference."""
    if any(np.isnan(v) for v in [mean_a, mean_b, effect_r]):
        return ""
    delta  = mean_b - mean_a
    higher = label_a if mean_a > mean_b else label_b
    lower  = label_b if mean_a > mean_b else label_a

    if metric == "offset_cents":
        sharp, flat, d = (label_a, label_b, -delta) if delta < 0 else (label_b, label_a, delta)
        if d < 5:
            return (f"The intonation of {svara} is nearly identical between groups "
                    f"(difference &lt; 5 cents). No meaningful śruti divergence.")
        elif d < 20:
            return (f"{sharp} sings {svara} approximately {d:.1f} cents sharper than {flat}. "
                    f"Differences of this scale are perceptible to trained listeners and may "
                    f"reflect a distinct śruti preference, school tradition (paramparā), or "
                    f"rāga-specific intonation practice.")
        else:
            return (f"A substantial divergence of {d:.1f} cents: {sharp} places {svara} "
                    f"markedly higher than {flat}. This is musicologically significant — "
                    f"differences above ~20 cents constitute a genuine difference in svara "
                    f"identity and are characteristic of distinct performance lineages or "
                    f"rāga interpretations.")

    elif metric == "peak_height":
        d_pct = 100.0 * abs(delta) / max(mean_a, mean_b)
        return (f"{higher} spends proportionally more time on {svara} "
                f"({d_pct:.0f}% higher relative density). "
                f"This suggests {svara} carries greater melodic weight in {higher}'s rendition — "
                f"consistent with a more prominent vādi, samvādi, or dīrgha svara function. "
                f"In {lower}'s performance, {svara} appears to fulfil a more transitional or "
                f"alpa role.")

    elif metric == "peak_width":
        wider    = label_a if mean_a > mean_b else label_b
        narrower = label_b if mean_a > mean_b else label_a
        w_wide   = max(mean_a, mean_b)
        w_narrow = min(mean_a, mean_b)
        return (f"{wider} shows greater pitch spread around {svara} "
                f"(FWHM {w_wide:.0f} vs {w_narrow:.0f} cents). "
                f"This is consistent with heavier or more elaborate gamaka usage — oscillatory, "
                f"sliding, or shaking ornaments that widen the pitch distribution. "
                f"{narrower} treats {svara} more steadily, suggesting either sparser ornamentation "
                f"or a more anchored, khaṇḍa-based style.")

    elif metric == "skewness":
        if abs(delta) < 0.15:
            return (f"Both groups approach {svara} with similar directional symmetry. "
                    f"No meaningful difference in gamaka direction.")
        pos_grp = label_a if mean_a > mean_b else label_b
        neg_grp = label_b if mean_a > mean_b else label_a
        return (f"{pos_grp} shows a more positively skewed pitch distribution around {svara}, "
                f"indicating a tendency to linger on pitches above the nominal position or "
                f"approach from below — characteristic of an ascending (ārohana) gamaka character. "
                f"{neg_grp} leans toward pitches below the nominal pitch, reflecting a more "
                f"descending (avarohana) approach or a downward meend/jāru tendency.")

    elif metric == "kurtosis":
        if abs(delta) < 0.2:
            return (f"Similar ornamental density on {svara} between groups — "
                    f"comparable degree of pitch concentration.")
        peaked = label_a if mean_a > mean_b else label_b
        spread = label_b if mean_a > mean_b else label_a
        return (f"{peaked} shows higher kurtosis on {svara}: pitch values cluster tightly around "
                f"the nominal position with occasional brief excursions. This is characteristic of "
                f"a sustained, anchored svara — possibly with sparse, precise gamaka. "
                f"{spread}'s flatter distribution suggests dense, continuous ornamentation such as "
                f"kampita (oscillation) or jāru (glide), where the pitch rarely settles on a "
                f"single point.")

    elif metric == "peak_loc":
        return (f"The modal pitch of {svara} differs by {abs(delta):.1f} cents between groups, "
                f"placing the characteristic intonation in a subtly different position within "
                f"the svara's tonal space.")

    return ""


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" />'


def _svara_axis(ax) -> None:
    for s in SVARA_GRID:
        ax.axvline(s, color="#cccccc", lw=0.8, ls="--", alpha=0.7, zorder=0)
    ax.set_xticks(SVARA_GRID)
    ax.set_xticklabels(SVARA_NAMES, fontsize=9)
    ax.set_xlim(0, 1200)


def fig_kde_overlay(all_kde: dict, all_cents: dict) -> object:
    """
    Mean ± 1 SD KDE for each group with faint raw-histogram background.
    Individual per-recording KDEs are shown as thin translucent traces.
    """
    _apply_plot_style()
    colors = _group_colors(list(all_kde.keys()))
    fig, ax = plt.subplots(figsize=(14, 4.5))

    # --- Faint background histogram (one per group, pooled across recordings) ---
    bin_edges = np.linspace(0, 1200, 121)   # 10-cent bins
    bin_w     = bin_edges[1] - bin_edges[0]
    bin_ctrs  = (bin_edges[:-1] + bin_edges[1:]) / 2
    for label, cents_list in sorted(all_cents.items()):
        pooled = np.concatenate(cents_list)
        counts, _ = np.histogram(pooled, bins=bin_edges)
        density   = counts / (counts.sum() * bin_w)
        ax.bar(bin_ctrs, density, width=bin_w * 0.9,
               color=colors[label], alpha=0.10, zorder=1)

    # --- Per-recording traces (thin, translucent) ---
    for label, kde_list in sorted(all_kde.items()):
        x = kde_list[0][0]
        for x_i, y_i in kde_list:
            ax.plot(x_i, y_i, color=colors[label], lw=0.6, alpha=0.18, zorder=2)

    # --- Mean ± 1 SD ---
    for label, kde_list in sorted(all_kde.items()):
        x   = kde_list[0][0]
        ys  = np.array([d[1] for d in kde_list])
        mu  = ys.mean(0)
        sig = ys.std(0)
        c   = colors[label]
        ax.fill_between(x, mu - sig, mu + sig, color=c, alpha=0.20, zorder=3)
        ax.plot(x, mu, color=c, lw=2.2, label=label, zorder=4)

    _svara_axis(ax)
    ax.set_xlabel("Cents from tonic")
    ax.set_ylabel("Density")
    ax.set_title("Pitch-class distribution by group  (mean KDE ± 1 SD · faint: individual recordings · bars: pooled histogram)")
    ax.legend()
    plt.tight_layout()
    return fig


def fig_boxplots(all_peaks: dict, omnibus: dict) -> dict:
    """One box-plot figure per metric; N box groups per svara position."""
    _apply_plot_style()
    labels   = sorted(all_peaks.keys())
    colors   = _group_colors(labels)
    n_groups = len(labels)
    spacing  = n_groups + 1.8

    figs = {}
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(14, 4 + 0.4 * n_groups))
        bps = []
        for g_idx, label in enumerate(labels):
            positions = np.arange(len(SVARA_NAMES)) * spacing + g_idx
            data = []
            for name in SVARA_NAMES:
                v = _collect_metric(all_peaks[label], name, metric)
                data.append(v if v else [float("nan")])
            bp = ax.boxplot(
                data, positions=positions, widths=0.75, patch_artist=True,
                boxprops=dict(facecolor=colors[label], alpha=0.70),
                medianprops=dict(color="#333333", lw=1.8),
                whiskerprops=dict(lw=0.9, color="#777777"),
                capprops=dict(lw=0.9, color="#777777"),
                flierprops=dict(marker="o", markersize=2, alpha=0.4,
                                markerfacecolor=colors[label]),
                showfliers=True,
            )
            bps.append((bp, label))

        all_finite = [
            v for lbl in labels for name in SVARA_NAMES
            for v in _collect_metric(all_peaks[lbl], name, metric)
        ]
        if all_finite:
            y_top  = max(all_finite)
            y_span = (y_top - min(all_finite)) or 1.0
            for i, name in enumerate(SVARA_NAMES):
                if omnibus[name][metric].get("significant", False):
                    center = i * spacing + (n_groups - 1) / 2
                    ax.text(center, y_top + y_span * 0.06, "✱",
                            ha="center", va="bottom", fontsize=13, color="#c0392b")

        center_ticks = np.arange(len(SVARA_NAMES)) * spacing + (n_groups - 1) / 2
        ax.set_xticks(center_ticks)
        ax.set_xticklabels(SVARA_NAMES, fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]}  (✱ = Kruskal-Wallis significant after Bonferroni correction)")
        ax.legend([bp["boxes"][0] for bp, _ in bps], [lbl for _, lbl in bps])
        plt.tight_layout()
        figs[metric] = fig
    return figs


def fig_presence(all_peaks: dict) -> object:
    """Grouped bar chart: fraction of recordings per group where each svara is present."""
    _apply_plot_style()
    labels   = sorted(all_peaks.keys())
    colors   = _group_colors(labels)
    n_groups = len(labels)
    x        = np.arange(len(SVARA_NAMES))
    width    = min(0.8 / n_groups, 0.35)
    offsets  = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * width

    fig, ax = plt.subplots(figsize=(13, 4))
    for label, offset in zip(labels, offsets):
        peaks_list = all_peaks[label]
        n = len(peaks_list)
        fracs = [
            sum(1 for p in peaks_list if p[name] is not None and p[name]["present"]) / n
            for name in SVARA_NAMES
        ]
        ax.bar(x + offset, fracs, width, label=label,
               color=colors[label], alpha=0.78, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(SVARA_NAMES, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Fraction of recordings")
    ax.set_title(
        f"Svara presence rate per group  "
        f"(threshold = {PRESENCE_FACTOR:.1f}× uniform baseline)"
    )
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_CSS = """\
body {
    font-family: "Helvetica Neue", Arial, sans-serif;
    max-width: 1240px;
    margin: 0 auto;
    padding: 2rem 2.5rem;
    color: #2c2c2c;
    line-height: 1.6;
    background: #ffffff;
}
h1 {
    font-weight: 300;
    font-size: 1.8rem;
    border-bottom: 2px solid #6baed6;
    padding-bottom: 0.5rem;
    color: #1a3a5c;
}
h2 {
    font-weight: 400;
    font-size: 1.2rem;
    border-bottom: 1px solid #d0d8e4;
    padding-bottom: 0.3rem;
    margin-top: 2.5rem;
    color: #1a3a5c;
}
h3 { margin-top: 1.8rem; color: #2d5a8a; font-weight: 500; font-size: 1rem; }
h4 { margin-top: 1.1rem; color: #555; font-size: 0.95rem; font-weight: 500; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.8rem 0 1.4rem 0;
    font-size: 0.82rem;
}
th, td { border: 1px solid #dde3ec; padding: 0.3rem 0.6rem; text-align: right; }
th     { background: #eef2f8; font-weight: 600; text-align: center; color: #2d4a70; }
td.lbl { text-align: left; font-weight: 600; color: #333; }
tr.sig { background: #fef9e7; }
tr.grp { background: #f4f7fb; }
tr:hover { background: #f0f4fb; }
.note  { color: #777; font-size: 0.79rem; font-style: italic; margin: 0.3rem 0; }
.warn  { background: #fff8e1; border-left: 3px solid #f9a825; padding: 0.5rem 0.8rem; font-size: 0.85rem; margin: 0.6rem 0; border-radius: 3px; }
.interp {
    font-size: 0.88rem;
    color: #3d3d5c;
    background: #f5f4ff;
    border-left: 3px solid #9e9ac8;
    padding: 0.5rem 0.8rem;
    margin: 0.4rem 0 0.9rem 0;
    border-radius: 0 4px 4px 0;
    line-height: 1.5;
}
img    { max-width: 100%; display: block; margin: 0.8rem 0 1.2rem 0; border-radius: 4px; }
section { margin-bottom: 3rem; }
.pill  {
    display: inline-block; background: #c0392b; color: #fff;
    border-radius: 3px; padding: 0 5px; font-size: 0.73rem;
    vertical-align: middle; margin-left: 4px;
}
details { margin: 0.3rem 0; }
details summary { cursor: pointer; color: #2d5a8a; font-size: 0.88rem; }
.svara-block { border: 1px solid #dde3ec; border-radius: 6px; padding: 1rem 1.2rem; margin-bottom: 1.5rem; }
.svara-block h3 { margin-top: 0; }
.tbl-label { font-size: 0.78rem; color: #666; margin: 0.6rem 0 0.2rem 0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
"""


def _fmt(v, d: int = 3) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "—"
    return f"{v:.{d}f}"


def _sig(row: dict) -> str:
    return '<span class="pill">sig</span>' if row.get("significant") else ""


def _section(html: str) -> str:
    return f"<section>{html}</section>"


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def build_report(
    all_paths: dict,         # {label: [path, ...]}
    all_kde:   dict,         # {label: [(x, y), ...]}
    all_cents: dict,         # {label: [cents_array, ...]}
    all_peaks: dict,         # {label: [svara_stats, ...]}
    omnibus:   dict,
    pairwise:  dict,
    n_peaks_omnibus:  dict,
    n_peaks_pairwise: dict,
    n_tests_omnibus:  int,
    n_tests_pairwise: int,
    alpha: float = 0.05,
) -> str:
    labels = sorted(all_peaks.keys())
    pairs  = list(combinations(labels, 2))

    corrected_alpha_omni = alpha / n_tests_omnibus
    corrected_alpha_pair = alpha / n_tests_pairwise if n_tests_pairwise else float("nan")

    # ---- Figures -----------------------------------------------------------
    ov_fig  = fig_kde_overlay(all_kde, all_cents)
    ov_b64  = _b64(ov_fig);  plt.close(ov_fig)

    pr_fig  = fig_presence(all_peaks)
    pr_b64  = _b64(pr_fig);  plt.close(pr_fig)

    bp_figs = fig_boxplots(all_peaks, omnibus)
    bp_b64  = {m: _b64(f) for m, f in bp_figs.items()}
    for f in bp_figs.values():
        plt.close(f)

    # ---- Helpers for tables ------------------------------------------------
    def _summary_cells(s: dict) -> str:
        return (
            f"<td>{_fmt(s['mean'])}</td>"
            f"<td>{_fmt(s['std'])}</td>"
            f"<td>{_fmt(s['median'])}</td>"
            f"<td>{s['n']}</td>"
        )

    def _summary_header(label: str) -> str:
        return (
            f"<th>Mean ({label})</th><th>SD ({label})</th>"
            f"<th>Median ({label})</th><th>n ({label})</th>"
        )

    # ---- Section 1: Dataset ------------------------------------------------
    rows_ds = "".join(
        f"<tr><td class='lbl'>{lbl}</td><td>{len(all_paths[lbl])}</td></tr>"
        for lbl in labels
    )
    file_lists = "".join(
        "<details><summary>{}: {} recording(s)</summary><ul>{}</ul></details>".format(
            lbl, len(all_paths[lbl]),
            "".join(f"<li>{os.path.basename(p)}</li>" for p in all_paths[lbl])
        )
        for lbl in labels
    )
    n_pair_str = f"{len(pairs)} pair(s)" if len(pairs) != 1 else "1 pair"
    small_groups = [lbl for lbl in labels if len(all_paths[lbl]) < 3]
    small_sample_warning = (
        "<p class='warn'>&#9888; <strong>Small-sample caution:</strong> "
        + ", ".join(small_groups)
        + (" has" if len(small_groups) == 1 else " have")
        + " fewer than 3 recordings. Statistical tests run but results have very low"
        " power and should be interpreted as exploratory only.</p>"
    ) if small_groups else ""
    sec1 = _section(
        "<h2>1. Dataset</h2>"
        f"<table><tr><th>Group</th><th>Recordings</th></tr>{rows_ds}</table>"
        + file_lists
        + small_sample_warning
        + f"<p class='note'>Groups: {len(labels)} · Pairs: {n_pair_str} · "
          f"Omnibus Bonferroni denominator: {n_tests_omnibus} · "
          f"Pairwise Bonferroni denominator: {n_tests_pairwise} · "
          f"Family-wise α = {alpha}.</p>"
        + f"<p class='note'>Omnibus corrected threshold: {corrected_alpha_omni:.5f} · "
          f"Pairwise corrected threshold: {_fmt(corrected_alpha_pair, 5)}.</p>"
        + f"<p class='note'>Effect size: rank-biserial r. "
          f"|r| &lt; 0.1 negligible · 0.1–0.3 small · 0.3–0.5 medium · &gt; 0.5 large.</p>"
    )

    # ---- Section 2: KDE overlay --------------------------------------------
    sec2 = _section(
        "<h2>2. Pitch-class distributions</h2>"
        "<p>Mean KDE across recordings in each group (shading = ±1 SD).</p>"
        + _img(ov_b64)
    )

    # ---- Section 3: Svara presence -----------------------------------------
    # Omnibus n_peaks table
    np_rows = "".join(
        f"<tr><td class='lbl'>{lbl}</td>"
        f"{_summary_cells(n_peaks_omnibus['summaries'][lbl])}</tr>"
        for lbl in labels
    )
    np_hdr = "<tr><th>Group</th><th>Mean</th><th>SD</th><th>Median</th><th>n</th></tr>"
    np_kw  = (
        f"Kruskal-Wallis H = {_fmt(n_peaks_omnibus['H'])}, "
        f"p (Bonferroni) = {_fmt(n_peaks_omnibus['p_bonferroni'], 4)}"
        + _sig(n_peaks_omnibus)
    )
    sec3 = _section(
        "<h2>3. Svara presence</h2>"
        + _img(pr_b64)
        + "<h3>Number of present svaras per recording</h3>"
        + f"<table>{np_hdr}{np_rows}</table>"
        + f"<p class='note'>{np_kw}</p>"
    )

    # ---- Section 4: Omnibus results summary --------------------------------
    omni_sig = [
        (sv, mt, omnibus[sv][mt])
        for sv in SVARA_NAMES for mt in METRICS
        if omnibus[sv][mt].get("significant", False)
    ]
    if omni_sig:
        hdr_cols = "".join(f"<th>{lbl}</th>" for lbl in labels)
        omni_rows = ""
        for sv, mt, row in omni_sig:
            means = "".join(
                f"<td>{_fmt(row['summaries'][lbl]['mean'])}</td>" for lbl in labels
            )
            omni_rows += (
                f"<tr class='sig'>"
                f"<td class='lbl'>{sv}</td><td>{METRIC_LABELS[mt]}</td>"
                f"{means}"
                f"<td>{_fmt(row['H'])}</td>"
                f"<td>{_fmt(row['p_bonferroni'], 4)}</td>"
                "</tr>"
            )
        omni_table = (
            f"<table><tr><th>Svara</th><th>Metric</th>"
            f"{hdr_cols}<th>H</th><th>p (Bonferroni)</th></tr>"
            f"{omni_rows}</table>"
        )
    else:
        omni_table = "<p>No omnibus significant differences after Bonferroni correction.</p>"

    sec4 = _section(
        "<h2>4. Omnibus test (Kruskal-Wallis)</h2>"
        "<p>Tests whether any group differs for each (svara, metric) combination.</p>"
        + omni_table
    )

    # ---- Section 5: Pairwise summaries with qualitative interpretation ------
    pair_sections = []
    for la, lb in pairs:
        pw = pairwise[(la, lb)]
        sig_rows = [
            (sv, mt, pw[sv][mt])
            for sv in SVARA_NAMES for mt in METRICS
            if pw[sv][mt].get("significant", False)
        ]
        if sig_rows:
            rows_html = "".join(
                f"<tr class='sig'>"
                f"<td class='lbl'>{sv}</td><td>{METRIC_LABELS[mt]}</td>"
                f"<td>{_fmt(row['summary_a']['mean'])}</td>"
                f"<td>{_fmt(row['summary_b']['mean'])}</td>"
                f"<td>{_fmt(row['summary_b']['mean'] - row['summary_a']['mean'])}</td>"
                f"<td>{_fmt(row['p_bonferroni'], 4)}</td>"
                f"<td>{_fmt(row['effect_r'])}</td>"
                "</tr>"
                for sv, mt, row in sig_rows
            )
            tbl = (
                "<table>"
                f"<tr><th>Svara</th><th>Metric</th>"
                f"<th>Mean {la}</th><th>Mean {lb}</th>"
                "<th>Δ (B−A)</th><th>p (Bonferroni)</th><th>Effect r</th></tr>"
                + rows_html + "</table>"
            )
            interps = "".join(
                f"<div class='interp'><strong>{sv} — {METRIC_LABELS[mt]}:</strong> "
                + _interpret(sv, mt,
                             row["summary_a"]["mean"], row["summary_b"]["mean"],
                             row["effect_r"], la, lb)
                + "</div>"
                for sv, mt, row in sig_rows
                if _interpret(sv, mt,
                              row["summary_a"]["mean"], row["summary_b"]["mean"],
                              row["effect_r"], la, lb)
            )
        else:
            tbl = "<p>No significant differences found for this pair after Bonferroni correction.</p>"
            interps = ""

        np_pw = n_peaks_pairwise[(la, lb)]
        np_line = (
            f"<p class='note'>Active svaras per recording — "
            f"{la}: {_fmt(np_pw['summary_a']['mean'])} ± {_fmt(np_pw['summary_a']['std'])}, "
            f"{lb}: {_fmt(np_pw['summary_b']['mean'])} ± {_fmt(np_pw['summary_b']['std'])} "
            f"(p Bonferroni = {_fmt(np_pw['p_bonferroni'], 4)}{_sig(np_pw)})</p>"
        )
        pair_sections.append(f"<h3>{la} vs {lb}</h3>{tbl}{interps}{np_line}")

    sec5 = _section(
        "<h2>5. Pairwise comparisons (Mann-Whitney U)</h2>"
        "<p>Post-hoc pairwise tests with a single Bonferroni denominator across all pairs "
        "and (svara, metric) combinations. Highlighted rows are significant. "
        "Coloured blocks provide a qualitative musicological interpretation of each significant finding.</p>"
        + "".join(pair_sections)
    )

    # ---- Section 6: Per-metric boxplots ------------------------------------
    bp_html = "".join(
        f"<h3>{METRIC_LABELS[m]}</h3>{_img(bp_b64[m])}"
        for m in METRICS
    )
    sec6 = _section(
        "<h2>6. Per-metric comparisons</h2>"
        "<p>Stars mark svaras with a significant Kruskal-Wallis result after Bonferroni correction.</p>"
        + bp_html
    )

    # ---- Section 7: Full per-svara tables ----------------------------------
    # Two clearly separated tables per svara:
    #   Table A — Omnibus (Kruskal-Wallis): one row per metric, columns = groups
    #   Table B — Pairwise (MWU): one row per (metric × pair), columns = pair stats

    omni_group_hdr = "".join(
        f"<th>Mean ({lbl})</th><th>SD ({lbl})</th>" for lbl in labels
    )
    omni_header = (
        f"<tr><th>Metric</th>{omni_group_hdr}"
        "<th>H statistic</th><th>p (Bonferroni)</th></tr>"
    )

    pair_header = (
        "<tr><th>Metric</th><th>Comparison</th>"
        "<th>Mean A</th><th>SD A</th><th>Mean B</th><th>SD B</th>"
        "<th>Δ (B−A)</th><th>p (Bonferroni)</th><th>Effect r</th></tr>"
    )

    svara_blocks = []
    for name in SVARA_NAMES:
        # Omnibus table rows
        omni_rows = ""
        for metric in METRICS:
            row  = omnibus[name][metric]
            cls  = " class='sig'" if row.get("significant") else " class='grp'"
            gcells = "".join(
                f"<td>{_fmt(row['summaries'][lbl]['mean'])}</td>"
                f"<td>{_fmt(row['summaries'][lbl]['std'])}</td>"
                for lbl in labels
            )
            omni_rows += (
                f"<tr{cls}>"
                f"<td class='lbl'>{METRIC_LABELS[metric]}</td>"
                f"{gcells}"
                f"<td>{_fmt(row['H'])}</td>"
                f"<td>{_fmt(row['p_bonferroni'], 4)}{_sig(row)}</td>"
                "</tr>"
            )

        # Pairwise table rows
        pair_rows = ""
        for metric in METRICS:
            for la, lb in pairs:
                row = pairwise[(la, lb)][name][metric]
                cls = " class='sig'" if row.get("significant") else ""
                delta = row["summary_b"]["mean"] - row["summary_a"]["mean"]
                pair_rows += (
                    f"<tr{cls}>"
                    f"<td class='lbl'>{METRIC_LABELS[metric]}</td>"
                    f"<td>{la} vs {lb}</td>"
                    f"<td>{_fmt(row['summary_a']['mean'])}</td>"
                    f"<td>{_fmt(row['summary_a']['std'])}</td>"
                    f"<td>{_fmt(row['summary_b']['mean'])}</td>"
                    f"<td>{_fmt(row['summary_b']['std'])}</td>"
                    f"<td>{_fmt(delta)}</td>"
                    f"<td>{_fmt(row['p_bonferroni'], 4)}{_sig(row)}</td>"
                    f"<td>{_fmt(row['effect_r'])}</td>"
                    "</tr>"
                )

        svara_blocks.append(
            f"<div class='svara-block'>"
            f"<h3>{name}</h3>"
            f"<p class='tbl-label'>Omnibus — Kruskal-Wallis across all groups</p>"
            f"<table>{omni_header}{omni_rows}</table>"
            f"<p class='tbl-label'>Pairwise post-hoc — Mann-Whitney U</p>"
            f"<table>{pair_header}{pair_rows}</table>"
            f"</div>"
        )

    sec7 = _section(
        "<h2>7. Full per-svara statistics</h2>"
        "<p>Each svara has two tables. "
        "<strong>Omnibus:</strong> one row per metric showing mean ± SD for every group, "
        "the Kruskal-Wallis H statistic, and Bonferroni-corrected p-value — answers "
        "<em>&ldquo;does any group differ on this metric?&rdquo;</em> "
        "<strong>Pairwise:</strong> one row per metric &times; comparison pair showing each group's "
        "mean &plusmn; SD, the signed difference &Delta; = B &minus; A, Bonferroni p, and effect size r &mdash; answers "
        "<em>&ldquo;which specific pair differs, and by how much?&rdquo;</em> "
        "Yellow rows are Bonferroni-significant.</p>"
        + "".join(svara_blocks)
    )

    return "\n".join([
        "<!DOCTYPE html><html lang='en'>",
        "<head><meta charset='UTF-8'>",
        f"<title>Intonation Report — {', '.join(labels)}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>Intonation Analysis: {' · '.join(labels)}</h1>",
        sec1, sec2, sec3, sec4, sec5, sec6, sec7,
        "</body></html>",
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Enable debug-level logging.")
@click.pass_context
def cli(ctx, verbose):
    """Pitch distribution analysis for intonation comparison."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _configure_logging(verbose)


@cli.command()
@click.argument("audio_paths", type=click.Path(exists=True), nargs=-1, required=True)
@click.pass_context
def extract(ctx, audio_paths):
    """Extract and cache pitch tracks. Tonic sidecar (_tonic.txt) must exist."""
    log.info("Extracting %d file(s)", len(audio_paths))
    for path in audio_paths:
        log.info("Processing: %s", path)
        _load_or_extract_pitch(path)
    log.info("extract complete")


@cli.command()
@click.argument("manifest", type=click.Path(exists=True))
@click.option("--groups", "-g", multiple=True,
              help="Subset of group labels to include (default: all groups in manifest).")
@click.option("--output", default="report.html", show_default=True)
@click.option("--alpha", default=0.05, show_default=True,
              help="Family-wise error rate for Bonferroni correction.")
@click.pass_context
def run(ctx, manifest, groups, output, alpha):
    """
    Full pipeline: process all groups in MANIFEST, compare them, write HTML report.

    \b
    Manifest CSV columns: path, group, tonic (tonic optional per row).
    Use --groups to restrict analysis to a subset of the manifest groups.
    Requires at least 2 groups.
    """
    t_start = time.time()
    log.info("Loading manifest: %s", manifest)
    all_groups = load_manifest(manifest)
    log.info("Manifest contains %d group(s): %s", len(all_groups), sorted(all_groups.keys()))

    if groups:
        missing = [g for g in groups if g not in all_groups]
        if missing:
            raise click.UsageError(
                f"Requested groups not in manifest: {missing}. "
                f"Available: {sorted(all_groups.keys())}"
            )
        selected = {g: all_groups[g] for g in groups}
    else:
        selected = all_groups

    if len(selected) < 2:
        raise click.UsageError(
            f"Need at least 2 groups; found: {sorted(selected.keys())}"
        )

    for lbl, entries in sorted(selected.items()):
        log.info("  %-20s  %d file(s)", lbl, len(entries))

    all_paths:  dict = {}
    all_kde:    dict = {}
    all_cents:  dict = {}
    all_peaks:  dict = {}

    for label, entries in sorted(selected.items()):
        log.info("--- Processing group '%s' (%d file(s)) ---", label, len(entries))
        kde_list, cents_list, peaks_list = [], [], []
        for i, entry in enumerate(entries, 1):
            log.info("[%d/%d] %s", i, len(entries), entry["path"])
            x, y, stats, cents = analyse_file(entry["path"], tonic=entry["tonic"])
            kde_list.append((x, y))
            cents_list.append(cents)
            peaks_list.append(stats)
        all_paths[label]  = [e["path"] for e in entries]
        all_kde[label]    = kde_list
        all_cents[label]  = cents_list
        all_peaks[label]  = peaks_list
        log.info("Group '%s' done", label)

    log.info("Running statistical comparison...")
    omnibus, pairwise, n_peaks_omni, n_peaks_pair, n_tests_omni, n_tests_pair = \
        compare_groups(all_peaks, alpha=alpha)

    n_sig_omni = sum(
        omnibus[sv][mt].get("significant", False)
        for sv in SVARA_NAMES for mt in METRICS
    )
    n_sig_pair = sum(
        pairwise[pair][sv][mt].get("significant", False)
        for pair in pairwise
        for sv in SVARA_NAMES for mt in METRICS
    )
    log.info("Omnibus significant:  %d / %d", n_sig_omni, n_tests_omni)
    log.info("Pairwise significant: %d / %d", n_sig_pair, n_tests_pair)

    log.info("Building report → %s", output)
    html = build_report(
        all_paths, all_kde, all_cents, all_peaks,
        omnibus, pairwise,
        n_peaks_omni, n_peaks_pair,
        n_tests_omni, n_tests_pair,
        alpha=alpha,
    )
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(output) / 1024
    log.info("Report written: %s (%.1f KB)", output, size_kb)
    log.info("Total time: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    cli()
