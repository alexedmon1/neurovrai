#!/usr/bin/env python3
"""
T1w/T2w Ratio Analysis HTML Report Generation.

Creates interactive HTML reports summarizing T1w/T2w ratio analysis results
including visualizations and group statistics.
"""

import base64
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_ratio_visualization(
    ratio_file: Path,
    title: str = "T1w/T2w Ratio"
) -> str:
    """
    Create tri-planar visualization of ratio map.

    Parameters
    ----------
    ratio_file : Path
        Path to ratio map NIfTI file
    title : str
        Title for the visualization

    Returns
    -------
    str
        Base64-encoded PNG image
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        img = nib.load(str(ratio_file))
        data = img.get_fdata()

        # Get center of mass for slicing
        valid_mask = data > 0
        if valid_mask.sum() == 0:
            return ""

        coords = np.array(np.where(valid_mask))
        center = coords.mean(axis=1).astype(int)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Get slices
        axial = data[:, :, center[2]]
        coronal = data[:, center[1], :]
        sagittal = data[center[0], :, :]

        # Plot
        vmin, vmax = np.percentile(data[valid_mask], [5, 95])

        axes[0].imshow(np.rot90(axial), cmap='hot', vmin=vmin, vmax=vmax)
        axes[0].set_title('Axial')
        axes[0].axis('off')

        axes[1].imshow(np.rot90(coronal), cmap='hot', vmin=vmin, vmax=vmax)
        axes[1].set_title('Coronal')
        axes[1].axis('off')

        im = axes[2].imshow(np.rot90(sagittal), cmap='hot', vmin=vmin, vmax=vmax)
        axes[2].set_title('Sagittal')
        axes[2].axis('off')

        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
        cbar.set_label('T1w/T2w Ratio')

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return img_base64

    except Exception as e:
        logger.warning(f"Failed to create visualization: {e}")
        return ""


def generate_t1t2ratio_html_report(
    t1t2ratio_dir: Path,
    output_file: Optional[Path] = None
) -> Path:
    """
    Generate comprehensive HTML report for T1w/T2w ratio analysis.

    Parameters
    ----------
    t1t2ratio_dir : Path
        Directory containing T1-T2-ratio analysis results
    output_file : Path, optional
        Output HTML file path. Default: {t1t2ratio_dir}/group/t1t2ratio_report.html

    Returns
    -------
    Path
        Path to generated HTML report
    """
    t1t2ratio_dir = Path(t1t2ratio_dir)
    group_dir = t1t2ratio_dir / 'group'

    if output_file is None:
        output_file = group_dir / 't1t2ratio_report.html'

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating T1-T2-ratio HTML report: {output_file}")

    # Load summary data
    summary_csv = group_dir / 't1t2ratio_summary.csv'
    if summary_csv.exists():
        summary_df = pd.read_csv(summary_csv)
    else:
        # Generate from individual metrics
        summary_data = []
        for subject_dir in sorted(t1t2ratio_dir.iterdir()):
            if not subject_dir.is_dir() or subject_dir.name in ['group', 'stats']:
                continue
            metrics_file = subject_dir / 't1t2ratio_metrics.json'
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    stats = metrics.get('statistics', {})
                    summary_data.append({
                        'subject': subject_dir.name,
                        'n_wm_voxels': stats.get('n_wm_voxels'),
                        'ratio_mean': stats.get('ratio_mean'),
                        'ratio_std': stats.get('ratio_std'),
                        'ratio_median': stats.get('ratio_median'),
                        'ratio_min': stats.get('ratio_min'),
                        'ratio_max': stats.get('ratio_max')
                    })
        summary_df = pd.DataFrame(summary_data)

    n_subjects = len(summary_df)
    mean_ratio = summary_df['ratio_mean'].mean() if 'ratio_mean' in summary_df else None
    std_ratio = summary_df['ratio_mean'].std() if 'ratio_mean' in summary_df else None

    # Create visualizations
    viz_html = ""
    merged_file = group_dir / 'merged_t1t2ratio.nii.gz'
    if merged_file.exists():
        viz_b64 = create_ratio_visualization(merged_file, "Group Mean T1w/T2w Ratio")
        if viz_b64:
            viz_html = f'<img src="data:image/png;base64,{viz_b64}" alt="Group visualization" style="max-width:100%;">'

    # Create distribution plot
    dist_html = ""
    if 'ratio_mean' in summary_df and len(summary_df) > 0:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Histogram of mean ratios
            axes[0].hist(summary_df['ratio_mean'].dropna(), bins=15, edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Mean T1w/T2w Ratio')
            axes[0].set_ylabel('Count')
            axes[0].set_title('Distribution of Mean Ratio')
            axes[0].axvline(summary_df['ratio_mean'].mean(), color='red', linestyle='--', label='Group Mean')
            axes[0].legend()

            # Box plot
            summary_df[['ratio_mean', 'ratio_std']].boxplot(ax=axes[1])
            axes[1].set_ylabel('Value')
            axes[1].set_title('Ratio Statistics')

            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            dist_b64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            dist_html = f'<img src="data:image/png;base64,{dist_b64}" alt="Distribution" style="max-width:100%;">'
        except Exception as e:
            logger.warning(f"Failed to create distribution plot: {e}")

    # Create subject table
    table_html = ""
    if not summary_df.empty:
        table_html = summary_df.to_html(
            index=False,
            classes='table table-striped table-hover',
            float_format=lambda x: f'{x:.4f}' if pd.notnull(x) else ''
        )

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>T1w/T2w Ratio Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .stat-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .table th, .table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .table th {{
            background-color: #667eea;
            color: white;
        }}
        .table tr:hover {{
            background-color: #f5f5f5;
        }}
        .methods {{
            background-color: #e8f4fd;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-top: 20px;
        }}
        .footer {{
            text-align: center;
            color: #666;
            padding: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>T1w/T2w Ratio Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="card">
        <h2>Summary Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{n_subjects}</div>
                <div class="stat-label">Subjects</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{f'{mean_ratio:.3f}' if mean_ratio else 'N/A'}</div>
                <div class="stat-label">Mean Ratio</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{f'{std_ratio:.3f}' if std_ratio else 'N/A'}</div>
                <div class="stat-label">Std Dev</div>
            </div>
        </div>
    </div>

    {'<div class="card"><h2>Group Visualization</h2>' + viz_html + '</div>' if viz_html else ''}

    {'<div class="card"><h2>Distribution</h2>' + dist_html + '</div>' if dist_html else ''}

    <div class="card">
        <h2>Subject Results</h2>
        {table_html}
    </div>

    <div class="card">
        <h2>Methods</h2>
        <div class="methods">
            <p><strong>T1w/T2w Ratio Computation:</strong></p>
            <ul>
                <li>T1w images: Bias-corrected using ANTs N4</li>
                <li>T2w images: Registered to T1w space using FSL FLIRT (6 DOF, correlation ratio)</li>
                <li>Ratio computation: T1w / T2w in native space</li>
                <li>Normalization: Applied to MNI152 space using T1w→MNI transform</li>
                <li>White matter masking: Threshold at 0.5 probability</li>
                <li>Smoothing: 4mm FWHM Gaussian kernel</li>
            </ul>
            <p><strong>Reference:</strong> Du G et al. (2019) Magnetic resonance T1w/T2w ratio:
            A parsimonious marker for Parkinson disease. Ann Neurol. 85(1):96-104.
            PMID: 30408230</p>
        </div>
    </div>

    <div class="footer">
        <p>Generated by neurovrai T1-T2-ratio analysis pipeline</p>
    </div>
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html_content)

    logger.info(f"Report generated: {output_file}")
    return output_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate T1-T2-ratio HTML report')
    parser.add_argument('t1t2ratio_dir', help='T1-T2-ratio analysis directory')
    parser.add_argument('--output', help='Output HTML file')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    report_path = generate_t1t2ratio_html_report(
        Path(args.t1t2ratio_dir),
        Path(args.output) if args.output else None
    )
    print(f"Report generated: {report_path}")
