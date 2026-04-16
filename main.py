import os
import click
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from compiam.melody.pitch_extraction import Melodia


SVARA_GRID = np.array(
    [0,100,200,300,400,500,600,
     700,800,900,1000,1100,1200],
    dtype=np.float64
)


def nearest_svara(value):
    idx = np.argmin(np.abs(SVARA_GRID - value))
    return SVARA_GRID[idx]


def compute_fwhm(x, y, peak_idx):
    """
    Full Width at Half Maximum (FWHM)
    """
    peak_height = y[peak_idx]
    half_max = peak_height / 2

    # search left
    left = peak_idx
    while left > 0 and y[left] > half_max:
        left -= 1

    # search right
    right = peak_idx
    while right < len(y)-1 and y[right] > half_max:
        right += 1

    return x[right] - x[left]


def hz_to_cents(freqs, tonic):
    freqs = np.asarray(freqs, dtype=np.float64)
    return 1200 * np.log2(freqs / tonic)


@click.group()
def cli():
    """Pitch extraction and histogram tools."""
    pass

@cli.command()
@click.argument("audio_paths", type=click.Path(exists=True), nargs=-1)
def extract(audio_paths):
    """
    Extract pitch tracks and tonics from one or more AUDIO_PATHS and save as TSV + TXT.
    """

    from compiam.melody.pitch_extraction import Melodia
    from compiam.melody.tonic_identification import TonicIndianMultiPitch

    melodia = Melodia()
    tonic_multipitch = TonicIndianMultiPitch()

    for audio_path in audio_paths:
        click.echo(f"Processing {audio_path}")

        # Extract pitch track
        pitch_track = melodia.extract(audio_path)
        root, _ = os.path.splitext(audio_path)
        pitch_track_path = root + ".tsv"
        tonic_path = root + "_tonic.txt"

        # Extract tonic
        click.echo("  Automatically extracting tonic")
        tonic = tonic_multipitch.extract(audio_path)

        # Save pitch track and tonic
        np.savetxt(pitch_track_path, pitch_track, delimiter="\t")
        np.savetxt(tonic_path, [tonic])
        click.echo(f"  Saved pitch track to {pitch_track_path}")
        click.echo(f"  Saved tonic to {tonic_path}")
    

@cli.command()
@click.argument("pitch_track_paths", type=click.Path(exists=True), nargs=-1)
@click.option("--bw-method", default=0.075, type=float)
@click.option("--output-prefix", default="combined")
def histogram(pitch_track_paths, bw_method, output_prefix):
    """
    Compute a single combined histogram + KDE from multiple pitch tracks,
    each with a corresponding tonic TXT file (derived automatically).
    """

    all_cents = []

    for pitch_path in pitch_track_paths:
        tonic_path = os.path.splitext(pitch_path)[0] + "_tonic.txt"

        click.echo(f"Loading {pitch_path} with tonic from {tonic_path}")

        pitch_track = np.loadtxt(pitch_path, delimiter="\t")
        tonic = float(np.loadtxt(tonic_path))

        pitch_vals = pitch_track[:, 1]
        pitch_vals = pitch_vals[pitch_vals > 0]

        if len(pitch_vals) == 0:
            click.echo(f"  No nonzero pitches in {pitch_path}, skipping.")
            continue

        pitch_vals_cents = hz_to_cents(pitch_vals, tonic) % 1200
        all_cents.append(pitch_vals_cents)

    if not all_cents:
        click.echo("No valid pitch values found, aborting.")
        return

    # Combine all tracks
    all_cents = np.concatenate(all_cents)

    # KDE
    kde = gaussian_kde(all_cents, bw_method=bw_method)
    x_range = np.linspace(0, 1200, 1000)
    kde_values = kde(x_range)

    # Histogram
    counts, bins = np.histogram(all_cents, bins=1000, range=(0,1200))
    centers = (bins[:-1] + bins[1:]) / 2
    density = counts / len(all_cents)
    combined_array = np.column_stack((centers, density))

    hist_path = f"{output_prefix}.dat"
    plot_path = f"{output_prefix}.png"

    np.savetxt(hist_path, combined_array, fmt="%.6f", delimiter=" ")
    click.echo(f"Saved combined histogram data to {hist_path}")

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(all_cents, bins=50, density=True, alpha=0.7)
    plt.plot(x_range, kde_values, linewidth=2)
    plt.xticks(
        [0,100,200,300,400,500,600,700,800,900,1000,1100,1200],
        ['S','R1','R2/G1','R3/G2','G3','M1','M2',
         'P','D1','D2/N1','N2/D3','N3','S']
    )
    plt.title(f"Kernel Density Estimation for combined tracks")
    plt.xlabel("Pitch (cents)")
    plt.ylabel("Density")
    plt.savefig(plot_path)
    plt.close()
    click.echo(f"Saved combined plot to {plot_path}")


@cli.command()
@click.argument("histogram_path", type=click.Path(exists=True))
def parameterise(histogram_path):
    """
    Parameterise histogram peaks by taking the nearest local peak to each svara.
    """

    click.echo(f"Parameterising histogram: {histogram_path}")

    data = np.loadtxt(histogram_path, delimiter=" ")
    centers = data[:, 0]
    density = data[:, 1]

    results = []
    window_width = 50  # search ±50 cents around each svara

    for svara in SVARA_GRID:  # exclude final 1200 (octave repeat)
        # create mask for window around the svara
        window_mask = (centers >= svara - window_width) & (centers <= svara + window_width)
        
        if not np.any(window_mask):
            continue

        local_density = density[window_mask]
        local_centers = centers[window_mask]
        
        # index of local maximum in this window
        idx_max = np.argmax(local_density)

        peak_loc = local_centers[idx_max]
        peak_height = local_density[idx_max]
        assigned = svara
        peak_idx = np.where(centers == peak_loc)[0][0]
        peak_width = compute_fwhm(centers, density, peak_idx)

        # Extract local region for shape statistics
        shape_mask = (centers > peak_loc - peak_width) & (centers < peak_loc + peak_width)
        local_vals = np.repeat(
            centers[shape_mask],
            (density[shape_mask] * 10000).astype(int)
        )

        if len(local_vals) > 10:
            peak_skew = skew(local_vals)
            peak_kurt = kurtosis(local_vals, fisher=True)
        else:
            peak_skew = np.nan
            peak_kurt = np.nan

        results.append([
            peak_loc,
            assigned,
            peak_height,
            peak_width,
            peak_skew,
            peak_kurt
        ])

    results = np.array(results)

    root, _ = os.path.splitext(histogram_path)
    param_path = root + "_params.tsv"

    header = "peak_loc\tassigned_svara\tpeak_height\tpeak_width\tskewness\texcess_kurtosis"

    np.savetxt(
        param_path,
        results,
        fmt="%.6f",
        delimiter="\t",
        header=header,
        comments=""
    )

    click.echo(f"Saved parameters to {param_path}")
if __name__ == "__main__":
    cli()