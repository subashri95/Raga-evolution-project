# Rāga Intonation Analysis

A pipeline for comparing svara intonation between two groups of Carnatic or Hindustānī recordings. For each recording it extracts the predominant pitch track, builds a tonic-normalised KDE histogram, and characterises each svara peak by location, height, width, skewness, and kurtosis. A Mann-Whitney U test with Bonferroni correction is run across all (svara, metric) pairs and the results are compiled into a self-contained HTML report.

## Install

```bash
mkvirtualenv raga --python=python3.9
workon raga
pip install -r requirements.txt
```

## Prepare your files

Pitch tracks are extracted automatically on first run and cached as `.tsv` files next to the audio. Subsequent runs load the cache directly.

## Manifest

Create a CSV with `path`, `group`, and `tonic` columns:

```csv
path,group,tonic
audio/groupA/recording_01.wav,GroupA,293.66
audio/groupA/recording_02.wav,GroupA,311.13
audio/groupB/recording_01.wav,GroupB,220.00
audio/groupB/recording_02.wav,GroupB,233.08
```

- **path**: absolute or relative path to the audio file
- **group**: label string — must match `--label-a` / `--label-b` exactly
- **tonic**: tonic frequency in Hz (e.g. `293.66`) **or** a note name string (e.g. `A`, `F#`, `Bb`). Note names are resolved to the third octave (C3 = 130.81 Hz). Enharmonic aliases (`Db`, `Eb`, `Gb`, `Ab`, `Bb`) are accepted.

The `tonic` column is optional. If it is absent or a cell is empty, the pipeline falls back to a `_tonic.txt` sidecar file at the same path as the audio (containing a single float in Hz). Having the tonic in the manifest is generally preferable as it keeps all metadata in one place.

## Commands

### Extract pitch tracks (optional pre-cache step)

```bash
python main.py extract audio/recording_01.wav audio/recording_02.wav
```

Cached `.tsv` files are reused automatically by the `run` command, so this step is optional.

### Run the full comparison pipeline

```bash
python main.py run manifest.csv --output report.html
```

All groups found in the manifest are included automatically. To restrict to a subset:

```bash
python main.py run manifest.csv --groups GroupA --groups GroupB --output report.html
```

Open `report.html` in any browser. The file is fully self-contained (all figures are base64-encoded inline).

To include focused hypothesis tests (see below):

```bash
python main.py run manifest.csv --hypotheses hypotheses.yaml --output report.html
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--groups` / `-g` | all groups | Repeat to restrict to a subset of manifest groups |
| `--output` | `report.html` | Output HTML path |
| `--alpha` | `0.05` | Family-wise error rate for Bonferroni correction |
| `--hypotheses` | *(none)* | Path to a YAML hypotheses file (optional) |

## Focused hypotheses

When you have specific musicological questions — e.g. *"is D1 more prominent in Group 1 than Group 2?"* — state them in a YAML file instead of relying on the exploratory sweep. This matters statistically: the Bonferroni correction divides by the number of hypotheses you actually state, not by 72 (12 svaras × 6 metrics). With 5 hypotheses the corrected threshold is 0.010 instead of 0.0007, giving substantially more power.

### Hypotheses file format

```yaml
hypotheses:
  - svara: D1
    metric: peak_height
    comparison: Group1 > Group2
    label: "D1 more prominent in Group 1"

  - svara: G3
    metric: offset_cents
    comparison: Group1 < Group2
    label: "G3 flatter in Group 1"

  - svara: M1
    metric: peak_width
    comparison: Group1 != Group2
    label: "M1 gamaka intensity differs"
```

| Field | Required | Values |
|-------|----------|--------|
| `svara` | yes | `S` `R1` `R2/G1` `R3/G2` `G3` `M1` `M2` `P` `D1` `D2/N1` `N2/D3` `N3` |
| `metric` | yes | `peak_loc` · `offset_cents` · `peak_height` · `peak_width` · `skewness` · `kurtosis` |
| `comparison` | yes | `GroupA > GroupB` · `GroupA < GroupB` · `GroupA != GroupB` |
| `label` | no | human-readable description (defaults to the comparison string) |

**Directional hypotheses** (`>` / `<`) use a one-sided Mann-Whitney U test, which halves the p-value relative to two-sided when the direction is correct. Only use these when you have a genuine prior prediction — stating a direction after seeing the data inflates the false-positive rate.

Results appear in **Section 8** of the report with their own Bonferroni-corrected p-values and effect sizes.

## Report structure

| Section | Contents |
|---------|----------|
| 1. Dataset | File counts, correction denominators, methodology notes |
| 2. Pitch-class distributions | Mean ± 1 SD KDE overlay for all groups |
| 3. Svara presence | Presence rate per svara; Kruskal-Wallis on count of present svaras |
| 4. Omnibus tests | Kruskal-Wallis per (svara, metric) — significant results highlighted |
| 5. Pairwise comparisons | MWU post-hoc for every pair of groups |
| 6. Per-metric boxplots | One figure per metric, all groups side by side |
| 7. Full per-svara tables | Omnibus + all pairwise results with effect sizes |
| 8. Focused hypotheses | *(only present when `--hypotheses` is supplied)* Per-hypothesis MWU results with focused Bonferroni correction |
| 9. Individual histograms | Collapsible — KDE + pitch histogram for every recording |

## Statistical approach

The pipeline scales to any number of groups N using a two-tier strategy:

**Tier 1 — Omnibus (Kruskal-Wallis):** Tests whether *any* group differs for each (svara, metric) combination. Bonferroni correction over 12 svaras × 6 metrics = 72 tests.

**Tier 2 — Pairwise post-hoc (Mann-Whitney U):** All N(N−1)/2 pairs are tested for every (svara, metric). A *single* Bonferroni denominator of N(N−1)/2 × 72 is applied across all pairwise tests simultaneously, preventing false positive inflation from running separate per-pair corrections.

- **Effect size**: rank-biserial correlation *r*. Guideline: |r| < 0.1 negligible · 0.1–0.3 small · 0.3–0.5 medium · > 0.5 large.
- **Skewness and kurtosis** are computed from raw pitch samples within ±50 cents of each svara (not from the KDE) to avoid bandwidth-induced smoothing bias.
- **Svara presence** (count of active svaras per recording) is tested with Kruskal-Wallis omnibus and MWU pairwise using an analogous correction.

## Histogram descriptors

Each svara peak is characterised by six descriptors. All six are computed independently per recording so that group-level statistics reflect genuine between-performer variation rather than averaging artefacts.

### Peak location (cents from tonic)

The position of the KDE mode within the ±50-cent window around each equal-temperament svara position. In Indian art music, svaras do not sit at fixed equal-temperament frequencies; each rāga prescribes its own characteristic intonation for each svara. A systematic shift in peak location between groups therefore reflects a difference in the characteristic intonation of that svara — for example, one group habitually singing a slightly flatter Gāndhāra. This is one of the most direct quantitative correlates of rāga grammar.

### Intonation offset from ET (cents)

The signed difference between the measured peak location and the equal-temperament position (peak location − ET position). Positive values indicate the svara is sung sharp of ET; negative values indicate flat. Reporting this offset separately from the raw peak location makes it easier to interpret small but musically meaningful deviations without reference to the absolute cent scale.

### Peak height (density)

The KDE density at the peak. Because the histogram is density-normalised (integrates to 1 over the octave), peak height represents the relative proportion of time the performer spends near that svara compared with all others. A high peak corresponds to a svara that is dwelt upon, emphasised, and ornamented at length. A low peak corresponds to a svara that is touched briefly in passing. Differences in peak height between groups reveal systematic differences in which svaras are emphasised, reflecting divergent interpretations of the rāga's hierarchy of notes.

### Peak width / FWHM (cents)

The full width at half maximum of the KDE peak. A narrow peak indicates that the performer settles steadily on the svara with little pitch deviation — characteristic of svaras that are held with minimal ornamentation. A wide peak indicates substantial pitch spread around the nominal svara position, most commonly caused by *gamaka*. Differences in peak width between groups therefore reflect differences in gamaka usage or gamaka intensity on a given svara.

### Skewness

The asymmetry of the pitch distribution within the svara window, computed from raw pitch samples. A positive skew means the distribution has a longer tail above the peak (the performer tends to arrive at the svara from below and overshoot, or dwell on pitches above it); negative skew indicates the opposite. Skewness encodes directional information about how the svara is approached. A svara whose gamaka consistently oscillates asymmetrically above or below its nominal position will produce a measurably skewed distribution. 

### Excess kurtosis

The peakedness (or flatness) of the pitch distribution relative to a Gaussian, computed from raw pitch samples (Fisher definition, so 0 = Gaussian). Positive excess kurtosis means pitch values cluster tightly around the svara with heavy tails — the performer spends most time precisely on the svara with occasional brief excursions. This is associated with a clean, sustained svara, possibly with sparse gamaka. Negative excess kurtosis indicates a flatter, more spread distribution — the performer rarely settles exactly on the nominal svara position, spending time spread across the window. This is the signature of dense, continuous gamaka such as *kampita* (oscillation), where the pitch is almost never stationary. Kurtosis is therefore a proxy for the degree and continuity of ornamentation on a svara.

## Statistical approach

- **Test**: Mann-Whitney U (two-sided, non-parametric — appropriate for small or non-Gaussian samples).
- **Correction**: Bonferroni over 12 svaras × 6 metrics = 72 tests (α ≈ 0.0007 at family-wise α = 0.05).
- **Effect size**: rank-biserial correlation *r*. Guideline: |r| < 0.1 negligible · 0.1–0.3 small · 0.3–0.5 medium · > 0.5 large.
- **Skewness and kurtosis** are computed from raw pitch samples within ±50 cents of each svara (not from the KDE) to avoid bandwidth-induced smoothing bias.
