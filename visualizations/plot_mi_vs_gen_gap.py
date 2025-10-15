"""
Visualization script for MI vs Generalization Gap analysis.

Creates scatter plots showing the relationship between generalization gap
and mutual information for different training ablations.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_results(results_dir: Path, architecture: str):
    """Load all result files for a given architecture.

    Returns:
        dict: Dictionary with ablation names as keys and lists of results as values
    """
    results = defaultdict(list)

    # Find all result files for this architecture
    pattern = f"{architecture}_*_results.npz"
    result_files = list(results_dir.glob(pattern))

    print(f"Found {len(result_files)} result files for {architecture}")

    for result_file in result_files:
        # Load the data
        data = np.load(result_file)

        # Extract ablation configuration from filename
        # Format: {arch}_{optimizer}_{batchnorm}_{augmentation}_seed{seed}_results.npz
        filename = result_file.stem  # Remove .npz

        # Remove architecture prefix and _results suffix
        config_str = filename.replace(f"{architecture}_", "").replace("_results", "")

        # Extract seed (always at the end: seed0, seed1, seed2)
        import re
        seed_match = re.search(r'seed(\d+)$', config_str)
        if not seed_match:
            print(f"Warning: Could not parse seed from {filename}")
            continue
        seed = seed_match.group(1)

        # Remove seed from config string
        config_str = config_str[:seed_match.start()].rstrip('_')

        # Now parse optimizer_batchnorm_augmentation
        # optimizer is always 'adam' or 'adamw' (first part)
        # batchnorm is 'bn' or 'no_bn'
        # augmentation is 'aug' or 'no_aug'
        if config_str.startswith('adamw_'):
            optimizer = 'adamw'
            remainder = config_str[6:]  # Remove 'adamw_'
        elif config_str.startswith('adam_'):
            optimizer = 'adam'
            remainder = config_str[5:]  # Remove 'adam_'
        else:
            print(f"Warning: Could not parse optimizer from {filename}")
            continue

        # Parse batchnorm and augmentation from remainder
        if remainder.startswith('no_bn_'):
            batchnorm = 'no_bn'
            augmentation = remainder[6:]  # Everything after 'no_bn_'
        elif remainder.startswith('bn_'):
            batchnorm = 'bn'
            augmentation = remainder[3:]  # Everything after 'bn_'
        else:
            print(f"Warning: Could not parse batchnorm from {filename}")
            continue

        ablation_name = f"{optimizer}_{batchnorm}_{augmentation}"

        # Extract values
        gen_gap = float(data['final_gen_gap'])
        mean_mi_masked = float(data['final_mean_mi_masked'])

        results[ablation_name].append({
            'gen_gap': gen_gap,
            'mean_mi_masked': mean_mi_masked,
            'seed': int(seed),
            'optimizer': optimizer,
            'batchnorm': batchnorm,
            'augmentation': augmentation
        })

    return results


def plot_mi_vs_gen_gap(results, architecture: str, output_dir: Path):
    """Create scatter plot of ln(10) - MI_masked vs generalization gap.

    Args:
        results: Dictionary of results by ablation
        architecture: Architecture name for title
        output_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors and markers for different ablations
    # We have 8 ablations: 2 optimizers × 2 batchnorm × 2 augmentation
    colors = {
        'adam': '#e74c3c',      # Red
        'adamw': '#3498db',     # Blue
    }

    markers = {
        'bn_aug': 'o',          # Circle
        'bn_no_aug': 's',       # Square
        'no_bn_aug': '^',       # Triangle up
        'no_bn_no_aug': 'D',    # Diamond
    }

    labels_added = set()

    # Calculate ln(10)
    ln_10 = np.log(10)

    # Plot each ablation
    for ablation_name, data_points in sorted(results.items()):
        if not data_points:
            continue

        # Extract data
        gen_gaps = [d['gen_gap'] for d in data_points]
        mean_mi_masked = [d['mean_mi_masked'] for d in data_points]

        # Calculate ln(10) - MI_masked
        y_values = [ln_10 - mi for mi in mean_mi_masked]

        # Get style parameters
        optimizer = data_points[0]['optimizer']
        batchnorm = data_points[0]['batchnorm']
        augmentation = data_points[0]['augmentation']

        color = colors[optimizer]
        marker_key = f"{batchnorm}_{augmentation}"
        marker = markers[marker_key]

        # Create label
        label = f"{optimizer.upper()}, {batchnorm.replace('_', ' ').title()}, {augmentation.replace('_', ' ').title()}"

        # Plot
        ax.scatter(gen_gaps, y_values,
                  c=color, marker=marker, s=150, alpha=0.7,
                  edgecolors='black', linewidths=1.5,
                  label=label)

    # Formatting
    ax.set_xlabel('Generalization Gap (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('ln(10) - MI_masked', fontsize=14, fontweight='bold')
    ax.set_title(f'Mutual Information vs Generalization Gap\n{architecture.upper()}',
                fontsize=16, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{architecture}_mi_vs_gen_gap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also save as PDF for publications
    output_path_pdf = output_dir / f"{architecture}_mi_vs_gen_gap.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Plot saved to: {output_path_pdf}")

    plt.close()


def print_summary(results):
    """Print summary statistics of the loaded results."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for ablation_name, data_points in sorted(results.items()):
        if not data_points:
            continue

        gen_gaps = [d['gen_gap'] for d in data_points]
        mean_mi_masked = [d['mean_mi_masked'] for d in data_points]

        ln_10 = np.log(10)
        y_values = [ln_10 - mi for mi in mean_mi_masked]

        print(f"\n{ablation_name}:")
        print(f"  N runs: {len(data_points)}")
        print(f"  Gen Gap: {np.mean(gen_gaps):.2f} ± {np.std(gen_gaps):.2f}%")
        print(f"  ln(10) - MI_masked: {np.mean(y_values):.4f} ± {np.std(y_values):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot MI vs Generalization Gap scatter plot'
    )

    parser.add_argument('--arch', type=str, default='vgg11',
                       choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'],
                       help='Architecture to plot (default: vgg11)')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory containing result files')
    parser.add_argument('--output_dir', type=str, default='./plots',
                       help='Directory to save plots')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    # Load results
    print(f"Loading results for {args.arch}...")
    results = load_results(results_dir, args.arch)

    if not results:
        print(f"No results found for {args.arch} in {results_dir}")
        return

    # Print summary
    print_summary(results)

    # Create plot
    print(f"\nCreating plot...")
    plot_mi_vs_gen_gap(results, args.arch, output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
