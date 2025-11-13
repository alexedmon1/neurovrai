#!/usr/bin/env python3
"""
Create HTML visualization of advanced DWI metrics and FA.

Generates false-colored images with proper colormaps for:
- FA (Fractional Anisotropy)
- DKI metrics (MK, AK, RK, KFA)
- NODDI metrics (FICVF, ODI, FISO)
- SANDI metrics (FSOMA, FNEURITE, FEC, RSOMA)
- ActiveAx metrics (FICVF, DIAM)
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import base64
from io import BytesIO
import sys

def create_png_from_nifti(nifti_file, slice_idx=None, colormap='hot', vmin=None, vmax=None, title=""):
    """Create PNG image from NIfTI file with specified colormap."""
    img = nib.load(nifti_file)
    data = img.get_fdata()

    # Get middle slice if not specified
    if slice_idx is None:
        slice_idx = data.shape[2] // 2

    # Extract slice
    slice_data = data[:, :, slice_idx]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(np.rot90(slice_data), cmap=colormap, vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)

    # Convert to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64


def generate_html_report(output_dir, output_html):
    """Generate HTML report with all available metrics."""

    output_dir = Path(output_dir)

    # Find available metrics
    metrics = []

    # FA
    fa_file = output_dir / 'dti' / 'FA.nii.gz'
    if fa_file.exists():
        metrics.append({
            'category': 'DTI',
            'name': 'FA',
            'file': fa_file,
            'colormap': 'hot',
            'vmin': 0,
            'vmax': 1,
            'description': 'Fractional Anisotropy [0-1]'
        })

    # DKI metrics
    dki_dir = output_dir / 'dki'
    if dki_dir.exists():
        dki_metrics = [
            ('mk', 'MK', 'viridis', 0, 3, 'Mean Kurtosis'),
            ('ak', 'AK', 'viridis', 0, 3, 'Axial Kurtosis'),
            ('rk', 'RK', 'viridis', 0, 3, 'Radial Kurtosis'),
            ('kfa', 'KFA', 'hot', 0, 1, 'Kurtosis Fractional Anisotropy'),
        ]
        for filename, name, cmap, vmin, vmax, desc in dki_metrics:
            file_path = dki_dir / f'{filename}.nii.gz'
            if file_path.exists():
                metrics.append({
                    'category': 'DKI',
                    'name': name,
                    'file': file_path,
                    'colormap': cmap,
                    'vmin': vmin,
                    'vmax': vmax,
                    'description': desc
                })

    # NODDI metrics
    noddi_dir = output_dir / 'noddi'
    if noddi_dir.exists():
        noddi_metrics = [
            ('ficvf', 'FICVF', 'hot', 0, 1, 'Neurite Density (Intracellular Volume Fraction) [0-1]'),
            ('odi', 'ODI', 'cool', 0, 1, 'Orientation Dispersion Index [0-1]'),
            ('fiso', 'FISO', 'Blues', 0, 1, 'Free Water Fraction [0-1]'),
        ]
        for filename, name, cmap, vmin, vmax, desc in noddi_metrics:
            file_path = noddi_dir / f'{filename}.nii.gz'
            if file_path.exists():
                metrics.append({
                    'category': 'NODDI',
                    'name': name,
                    'file': file_path,
                    'colormap': cmap,
                    'vmin': vmin,
                    'vmax': vmax,
                    'description': desc
                })

    # SANDI metrics
    sandi_dir = output_dir / 'sandi'
    if sandi_dir.exists():
        sandi_metrics = [
            ('fsoma', 'FSOMA', 'Purples', 0, 1, 'Soma Volume Fraction [0-1]'),
            ('fneurite', 'FNEURITE', 'Oranges', 0, 1, 'Neurite Volume Fraction [0-1]'),
            ('fec', 'FEC', 'Greens', 0, 1, 'Extra-cellular Fraction [0-1]'),
            ('fcsf', 'FCSF', 'Blues', 0, 1, 'CSF Fraction [0-1]'),
            ('rsoma', 'RSOMA', 'plasma', 1, 12, 'Soma Radius [μm]'),
        ]
        for filename, name, cmap, vmin, vmax, desc in sandi_metrics:
            file_path = sandi_dir / f'{filename}.nii.gz'
            if file_path.exists():
                metrics.append({
                    'category': 'SANDI',
                    'name': name,
                    'file': file_path,
                    'colormap': cmap,
                    'vmin': vmin,
                    'vmax': vmax,
                    'description': desc
                })

    # ActiveAx metrics
    activeax_dir = output_dir / 'activeax'
    if activeax_dir.exists():
        activeax_metrics = [
            ('ficvf', 'FICVF', 'hot', 0, 1, 'Intra-axonal Volume Fraction [0-1]'),
            ('diam', 'DIAM', 'plasma', 0.5, 5, 'Mean Axon Diameter [μm]'),
        ]
        for filename, name, cmap, vmin, vmax, desc in activeax_metrics:
            file_path = activeax_dir / f'{filename}.nii.gz'
            if file_path.exists():
                metrics.append({
                    'category': 'ActiveAx',
                    'name': name,
                    'file': file_path,
                    'colormap': cmap,
                    'vmin': vmin,
                    'vmax': vmax,
                    'description': desc
                })

    if not metrics:
        print("No metrics found!")
        return

    print(f"Found {len(metrics)} metrics to visualize")

    # Generate images
    print("Generating images...")
    for metric in metrics:
        print(f"  Processing {metric['category']}/{metric['name']}...")
        metric['image'] = create_png_from_nifti(
            metric['file'],
            colormap=metric['colormap'],
            vmin=metric['vmin'],
            vmax=metric['vmax'],
            title=f"{metric['category']}: {metric['name']}"
        )

    # Generate HTML
    print(f"Creating HTML report: {output_html}")

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Advanced DWI Metrics Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #666;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
            margin-top: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-card img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .metric-card .description {{
            margin-top: 10px;
            color: #666;
            font-size: 14px;
            text-align: center;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .timestamp {{
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <h1>Advanced DWI Metrics Visualization</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Output Directory:</strong> {output_dir}</p>
        <p><strong>Total Metrics:</strong> {len(metrics)}</p>
        <ul>
"""

    # Add summary by category
    categories = {}
    for metric in metrics:
        cat = metric['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(metric['name'])

    for cat, names in sorted(categories.items()):
        html_content += f"            <li><strong>{cat}:</strong> {', '.join(names)}</li>\n"

    html_content += """
        </ul>
    </div>
"""

    # Add metrics grouped by category
    for category in sorted(categories.keys()):
        html_content += f'\n    <h2>{category}</h2>\n'
        html_content += '    <div class="metrics-grid">\n'

        for metric in [m for m in metrics if m['category'] == category]:
            html_content += f"""
        <div class="metric-card">
            <img src="data:image/png;base64,{metric['image']}" alt="{metric['name']}">
            <div class="description">{metric['description']}</div>
        </div>
"""

        html_content += '    </div>\n'

    # Add footer
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content += f"""
    <div class="timestamp">
        Generated on {timestamp}
    </div>
</body>
</html>
"""

    # Write HTML file
    with open(output_html, 'w') as f:
        f.write(html_content)

    print(f"\n✓ HTML report created: {output_html}")
    print(f"  Open with: firefox {output_html}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path('/mnt/bytopia/development/IRC805/derivatives/dwi_topup/IRC805-0580101/advanced_models_amico')

    output_html = Path('dwi_metrics_visualization.html')

    generate_html_report(output_dir, output_html)
