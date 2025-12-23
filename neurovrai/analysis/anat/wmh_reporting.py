#!/usr/bin/env python3
"""
White Matter Hyperintensity (WMH) Reporting Module

Generates HTML reports and visualizations for WMH analysis results.

Features:
- Group summary statistics table
- WMH volume distribution histogram
- Tract-wise lesion burden heatmap
- Individual subject summaries
- Tri-planar lesion visualizations
"""

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from typing import Dict, List, Optional
import json
import base64
from io import BytesIO
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 encoded PNG."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def create_wmh_overlay_visualization(
    t2w_mni: Path,
    wmh_mask: Path,
    title: str = "WMH Overlay"
) -> str:
    """
    Create tri-planar visualization of WMH overlay on T2w.

    Parameters
    ----------
    t2w_mni : Path
        T2w image in MNI space
    wmh_mask : Path
        Binary WMH mask
    title : str
        Title for the visualization

    Returns
    -------
    str
        Base64 encoded PNG image
    """
    t2w_img = nib.load(t2w_mni)
    t2w_data = t2w_img.get_fdata()

    wmh_img = nib.load(wmh_mask)
    wmh_data = wmh_img.get_fdata()

    # Get center of mass of WMH or use image center
    if np.sum(wmh_data) > 0:
        coords = np.array(np.where(wmh_data > 0))
        center = coords.mean(axis=1).astype(int)
    else:
        center = np.array(t2w_data.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=14)

    # Axial (z)
    ax = axes[0]
    ax.imshow(t2w_data[:, :, center[2]].T, cmap='gray', origin='lower')
    ax.imshow(wmh_data[:, :, center[2]].T, cmap='Reds', alpha=0.5, origin='lower',
              vmin=0, vmax=1)
    ax.set_title(f'Axial (z={center[2]})')
    ax.axis('off')

    # Coronal (y)
    ax = axes[1]
    ax.imshow(t2w_data[:, center[1], :].T, cmap='gray', origin='lower')
    ax.imshow(wmh_data[:, center[1], :].T, cmap='Reds', alpha=0.5, origin='lower',
              vmin=0, vmax=1)
    ax.set_title(f'Coronal (y={center[1]})')
    ax.axis('off')

    # Sagittal (x)
    ax = axes[2]
    ax.imshow(t2w_data[center[0], :, :].T, cmap='gray', origin='lower')
    ax.imshow(wmh_data[center[0], :, :].T, cmap='Reds', alpha=0.5, origin='lower',
              vmin=0, vmax=1)
    ax.set_title(f'Sagittal (x={center[0]})')
    ax.axis('off')

    plt.tight_layout()
    return fig_to_base64(fig)


def create_volume_distribution_plot(df: pd.DataFrame) -> str:
    """
    Create histogram of WMH volumes across subjects.

    Parameters
    ----------
    df : pd.DataFrame
        Group summary DataFrame with 'total_volume_mm3' column

    Returns
    -------
    str
        Base64 encoded PNG image
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Volume distribution
    ax = axes[0]
    ax.hist(df['total_volume_mm3'], bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Total WMH Volume (mm³)')
    ax.set_ylabel('Number of Subjects')
    ax.set_title('WMH Volume Distribution')

    # Lesion count distribution
    ax = axes[1]
    ax.hist(df['n_lesions'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Number of Lesions')
    ax.set_ylabel('Number of Subjects')
    ax.set_title('Lesion Count Distribution')

    plt.tight_layout()
    return fig_to_base64(fig)


def create_tract_heatmap(tract_csv: Path) -> str:
    """
    Create heatmap of WMH burden by tract.

    Parameters
    ----------
    tract_csv : Path
        Path to tract summary CSV (wmh_by_tract.csv)

    Returns
    -------
    str
        Base64 encoded PNG image
    """
    df = pd.read_csv(tract_csv)

    # Sort by mean WMH volume
    df = df.sort_values('wmh_volume_mm3_mean', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))

    # Horizontal bar chart
    y_pos = range(len(df))
    ax.barh(y_pos, df['wmh_volume_mm3_mean'], xerr=df['wmh_volume_mm3_std'],
            align='center', alpha=0.7, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['tract_name'], fontsize=8)
    ax.set_xlabel('Mean WMH Volume (mm³)')
    ax.set_title('WMH Burden by White Matter Tract')

    plt.tight_layout()
    return fig_to_base64(fig)


def create_subject_scatter_plot(df: pd.DataFrame) -> str:
    """
    Create scatter plot of lesion count vs volume per subject.

    Parameters
    ----------
    df : pd.DataFrame
        Group summary DataFrame

    Returns
    -------
    str
        Base64 encoded PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(df['n_lesions'], df['total_volume_mm3'], alpha=0.7, s=100)

    # Add subject labels
    for _, row in df.iterrows():
        ax.annotate(row['subject'].replace('IRC805-', ''),
                   (row['n_lesions'], row['total_volume_mm3']),
                   fontsize=8, alpha=0.7)

    ax.set_xlabel('Number of Lesions')
    ax.set_ylabel('Total WMH Volume (mm³)')
    ax.set_title('Lesion Count vs Volume by Subject')

    plt.tight_layout()
    return fig_to_base64(fig)


def generate_wmh_html_report(
    hyperintensities_dir: Path,
    output_file: Optional[Path] = None
) -> Path:
    """
    Generate comprehensive HTML report for WMH analysis.

    Parameters
    ----------
    hyperintensities_dir : Path
        Root hyperintensities output directory
    output_file : Path, optional
        Output HTML file (default: {dir}/group/wmh_report.html)

    Returns
    -------
    Path
        Path to generated HTML report
    """
    hyperintensities_dir = Path(hyperintensities_dir)
    group_dir = hyperintensities_dir / 'group'

    if output_file is None:
        output_file = group_dir / 'wmh_report.html'

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load group summary
    summary_csv = group_dir / 'wmh_summary.csv'
    if not summary_csv.exists():
        logger.error(f"Group summary not found: {summary_csv}")
        return output_file

    df = pd.read_csv(summary_csv)

    # Calculate statistics
    n_subjects = len(df)
    mean_volume = df['total_volume_mm3'].mean()
    std_volume = df['total_volume_mm3'].std()
    mean_lesions = df['n_lesions'].mean()
    std_lesions = df['n_lesions'].std()

    # Generate visualizations
    logger.info("Generating visualizations...")

    volume_dist_img = create_volume_distribution_plot(df)
    scatter_img = create_subject_scatter_plot(df)

    tract_csv = group_dir / 'wmh_by_tract.csv'
    tract_heatmap_img = None
    if tract_csv.exists():
        tract_heatmap_img = create_tract_heatmap(tract_csv)

    # Generate example overlay (first subject with lesions)
    example_overlay_img = None
    for _, row in df.iterrows():
        if row['n_lesions'] > 0:
            subj_dir = hyperintensities_dir / row['subject']
            t2w_mni = subj_dir / 't2w_mni.nii.gz'
            wmh_mask = subj_dir / 'wmh_mask.nii.gz'
            if t2w_mni.exists() and wmh_mask.exists():
                try:
                    example_overlay_img = create_wmh_overlay_visualization(
                        t2w_mni, wmh_mask, f"Example: {row['subject']}"
                    )
                    break
                except Exception as e:
                    logger.warning(f"Could not create overlay for {row['subject']}: {e}")

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WMH Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #444;
            margin-top: 30px;
        }}
        .summary-box {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .stat-card .label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .visualization {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>White Matter Hyperintensity Analysis Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="summary-box">
        <h2>Summary Statistics</h2>
        <div class="stat-grid">
            <div class="stat-card">
                <div class="value">{n_subjects}</div>
                <div class="label">Subjects Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="value">{mean_lesions:.1f} &pm; {std_lesions:.1f}</div>
                <div class="label">Mean Lesion Count</div>
            </div>
            <div class="stat-card">
                <div class="value">{mean_volume:.1f}</div>
                <div class="label">Mean WMH Volume (mm³)</div>
            </div>
            <div class="stat-card">
                <div class="value">{std_volume:.1f}</div>
                <div class="label">SD WMH Volume (mm³)</div>
            </div>
        </div>
    </div>
"""

    if example_overlay_img:
        html_content += f"""
    <div class="visualization">
        <h2>Example WMH Overlay</h2>
        <img src="data:image/png;base64,{example_overlay_img}" alt="WMH Overlay">
        <p>Red overlay shows detected white matter hyperintensities</p>
    </div>
"""

    html_content += f"""
    <div class="visualization">
        <h2>Distribution of WMH Burden</h2>
        <img src="data:image/png;base64,{volume_dist_img}" alt="Volume Distribution">
    </div>

    <div class="visualization">
        <h2>Lesion Count vs Volume</h2>
        <img src="data:image/png;base64,{scatter_img}" alt="Scatter Plot">
    </div>
"""

    if tract_heatmap_img:
        html_content += f"""
    <div class="visualization">
        <h2>WMH Burden by White Matter Tract</h2>
        <img src="data:image/png;base64,{tract_heatmap_img}" alt="Tract Heatmap">
        <p>Error bars show standard deviation across subjects</p>
    </div>
"""

    # Subject table
    html_content += """
    <div class="summary-box">
        <h2>Individual Subject Results</h2>
        <table>
            <tr>
                <th>Subject</th>
                <th>Lesion Count</th>
                <th>Total Volume (mm³)</th>
                <th>Mean Lesion Size (mm³)</th>
                <th>Max Lesion Size (mm³)</th>
            </tr>
"""

    for _, row in df.sort_values('total_volume_mm3', ascending=False).iterrows():
        html_content += f"""
            <tr>
                <td>{row['subject']}</td>
                <td>{row['n_lesions']}</td>
                <td>{row['total_volume_mm3']:.2f}</td>
                <td>{row['mean_lesion_volume_mm3']:.2f}</td>
                <td>{row['max_lesion_volume_mm3']:.2f}</td>
            </tr>
"""

    html_content += """
        </table>
    </div>

    <div class="summary-box">
        <h2>Methods</h2>
        <p><strong>WMH Detection Algorithm:</strong></p>
        <ol>
            <li>T2w images co-registered to T1w space using FSL FLIRT (6 DOF, correlation ratio)</li>
            <li>T2w and WM mask normalized to MNI152 2mm space using ANTs</li>
            <li>Intensity threshold calculated as: mean + 2.5 × SD of T2w within WM mask</li>
            <li>Connected component labeling with 3-voxel minimum cluster size</li>
            <li>Tract-wise analysis using JHU ICBM-DTI-81 white matter atlas</li>
        </ol>
        <p><strong>Atlas:</strong> JHU ICBM-DTI-81 (48 white matter regions + 20 major tracts)</p>
    </div>

</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html_content)

    logger.info(f"Generated HTML report: {output_file}")
    return output_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate WMH HTML Report')
    parser.add_argument('--hyperintensities-dir', required=True, type=Path,
                       help='Hyperintensities directory')
    parser.add_argument('--output', type=Path, help='Output HTML file')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_wmh_html_report(args.hyperintensities_dir, args.output)
