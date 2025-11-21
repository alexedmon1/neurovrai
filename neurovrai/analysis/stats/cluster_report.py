#!/usr/bin/env python3
"""
Cluster Extraction and Reporting for FSL Randomise Results

Extracts significant clusters from randomise corrected p-value maps and
generates comprehensive reports with:
- Cluster size and peak coordinates
- Atlas labels for anatomical localization
- CSV and HTML formatted reports
- Summary statistics

Uses FSL's cluster command for cluster extraction.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import nibabel as nib
import numpy as np


class ClusterReportError(Exception):
    """Raised when cluster extraction fails"""
    pass


def run_fsl_cluster(
    stat_file: Path,
    corrp_file: Path,
    threshold: float = 0.95,
    min_cluster_size: int = 10,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Run FSL cluster command to extract significant clusters

    Args:
        stat_file: T-statistic or F-statistic map
        corrp_file: Corrected p-value map (1-p values from randomise)
        threshold: Threshold for significance (default: 0.95 = p<0.05)
        min_cluster_size: Minimum cluster size in voxels
        output_dir: Optional directory to save cluster outputs

    Returns:
        DataFrame with cluster information

    Raises:
        ClusterReportError: If cluster extraction fails
    """
    if not stat_file.exists():
        raise ClusterReportError(f"Stat file not found: {stat_file}")
    if not corrp_file.exists():
        raise ClusterReportError(f"Corrected p-value file not found: {corrp_file}")

    # Load corrected p-value map
    corrp_img = nib.load(corrp_file)
    corrp_data = corrp_img.get_fdata()

    # Create binary mask of significant voxels
    sig_mask = (corrp_data >= threshold).astype(np.uint8)
    n_sig_voxels = np.sum(sig_mask)

    if n_sig_voxels == 0:
        logging.info(f"No significant voxels found at threshold {threshold}")
        return pd.DataFrame()

    logging.info(f"Found {n_sig_voxels} significant voxels")

    # Save thresholded mask temporarily
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        thresh_file = output_dir / "thresh_mask.nii.gz"
    else:
        import tempfile
        temp_dir = tempfile.mkdtemp()
        thresh_file = Path(temp_dir) / "thresh_mask.nii.gz"

    thresh_img = nib.Nifti1Image(sig_mask, corrp_img.affine, corrp_img.header)
    nib.save(thresh_img, thresh_file)

    # Run FSL cluster command
    cluster_output = thresh_file.parent / "clusters.txt"

    cmd = [
        'cluster',
        '--in=' + str(stat_file),
        '--thresh=0',  # Already thresholded via mask
        '--mm',  # Output coordinates in mm
        '--min=' + str(min_cluster_size)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Parse cluster output
        if result.stdout.strip():
            # Write to file
            with open(cluster_output, 'w') as f:
                f.write(result.stdout)

            # Parse into DataFrame
            df = parse_cluster_output(result.stdout)
            logging.info(f"Extracted {len(df)} clusters")
            return df
        else:
            logging.info("No clusters found")
            return pd.DataFrame()

    except subprocess.CalledProcessError as e:
        logging.error(f"FSL cluster command failed: {e}")
        logging.error(f"stderr: {e.stderr}")
        raise ClusterReportError(f"Cluster extraction failed: {e}")


def parse_cluster_output(cluster_output: str) -> pd.DataFrame:
    """
    Parse FSL cluster command output into DataFrame

    Args:
        cluster_output: stdout from FSL cluster command

    Returns:
        DataFrame with cluster information
    """
    lines = cluster_output.strip().split('\n')

    # Find header line
    header_idx = None
    for i, line in enumerate(lines):
        if 'Cluster Index' in line or 'Voxels' in line:
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame()

    # Parse header
    header_line = lines[header_idx]
    headers = header_line.split()

    # Parse data lines
    data = []
    for line in lines[header_idx + 1:]:
        if line.strip():
            values = line.split()
            if len(values) == len(headers):
                data.append(values)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=headers)

    # Convert numeric columns
    numeric_cols = ['Voxels', 'MAX', 'MAX X (mm)', 'MAX Y (mm)', 'MAX Z (mm)',
                    'COG X (mm)', 'COG Y (mm)', 'COG Z (mm)']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def add_anatomical_labels(
    df: pd.DataFrame,
    atlas: str = 'JHU-ICBM-labels-1mm'
) -> pd.DataFrame:
    """
    Add anatomical labels to clusters using FSL atlas

    Args:
        df: DataFrame with cluster coordinates
        atlas: FSL atlas name (default: JHU-ICBM for white matter tracts)

    Returns:
        DataFrame with added 'Region' column
    """
    if df.empty:
        return df

    # Use atlasquery to get labels
    labels = []

    for _, row in df.iterrows():
        if 'MAX X (mm)' in row and 'MAX Y (mm)' in row and 'MAX Z (mm)' in row:
            x, y, z = row['MAX X (mm)'], row['MAX Y (mm)'], row['MAX Z (mm)']

            try:
                cmd = [
                    'atlasquery',
                    '-a', atlas,
                    '-c', f"{x},{y},{z}"
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False
                )

                if result.returncode == 0 and result.stdout.strip():
                    # Parse atlas output
                    label = parse_atlas_output(result.stdout)
                    labels.append(label)
                else:
                    labels.append('Unknown')

            except Exception as e:
                logging.warning(f"Failed to query atlas for coordinates ({x}, {y}, {z}): {e}")
                labels.append('Unknown')
        else:
            labels.append('Unknown')

    df['Region'] = labels
    return df


def parse_atlas_output(atlas_output: str) -> str:
    """
    Parse FSL atlasquery output to extract most likely region

    Args:
        atlas_output: Output from atlasquery command

    Returns:
        Region name or 'Unknown'
    """
    # Example output: "50% Right Corticospinal Tract<br>30% Right Superior Longitudinal Fasciculus"
    if not atlas_output.strip():
        return 'Unknown'

    # Split by <br> and take first (highest probability) region
    regions = atlas_output.split('<br>')

    if regions:
        # Extract region name (after percentage)
        first_region = regions[0].strip()
        parts = first_region.split('%', 1)
        if len(parts) > 1:
            return parts[1].strip()

    return 'Unknown'


def generate_cluster_report(
    stat_file: Path,
    corrp_file: Path,
    contrast_name: str,
    output_dir: Path,
    threshold: float = 0.95,
    min_cluster_size: int = 10,
    add_atlas_labels: bool = True,
    atlas: str = 'JHU-ICBM-labels-1mm'
) -> Dict:
    """
    Generate comprehensive cluster report

    Args:
        stat_file: T-statistic or F-statistic map
        corrp_file: Corrected p-value map
        contrast_name: Name of contrast for report
        output_dir: Output directory for reports
        threshold: Significance threshold (default: 0.95 = p<0.05)
        min_cluster_size: Minimum cluster size
        add_atlas_labels: Whether to add anatomical labels
        atlas: FSL atlas to use for labeling

    Returns:
        Dictionary with report paths and summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"\nGenerating cluster report for: {contrast_name}")
    logging.info(f"  Threshold: {threshold} (p < {1-threshold})")
    logging.info(f"  Min cluster size: {min_cluster_size} voxels")

    # Extract clusters
    df = run_fsl_cluster(
        stat_file=stat_file,
        corrp_file=corrp_file,
        threshold=threshold,
        min_cluster_size=min_cluster_size,
        output_dir=output_dir
    )

    if df.empty:
        logging.info(f"  No significant clusters found")
        return {
            'n_clusters': 0,
            'significant': False
        }

    # Add anatomical labels if requested
    if add_atlas_labels:
        logging.info(f"  Adding anatomical labels from {atlas}...")
        df = add_anatomical_labels(df, atlas=atlas)

    # Save CSV report
    csv_file = output_dir / f"{contrast_name}_clusters.csv"
    df.to_csv(csv_file, index=False)
    logging.info(f"  Saved CSV: {csv_file}")

    # Generate HTML report
    html_file = output_dir / f"{contrast_name}_clusters.html"
    generate_html_report(df, contrast_name, html_file, threshold)
    logging.info(f"  Saved HTML: {html_file}")

    # Summary
    total_voxels = df['Voxels'].sum() if 'Voxels' in df.columns else 0

    return {
        'n_clusters': len(df),
        'total_voxels': int(total_voxels),
        'significant': True,
        'csv_file': str(csv_file),
        'html_file': str(html_file)
    }


def generate_html_report(
    df: pd.DataFrame,
    contrast_name: str,
    output_file: Path,
    threshold: float
):
    """
    Generate HTML report for clusters

    Args:
        df: DataFrame with cluster information
        contrast_name: Name of contrast
        output_file: Output HTML file path
        threshold: Significance threshold used
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cluster Report: {contrast_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #e7f3fe; padding: 10px; border-left: 4px solid #2196F3; }}
    </style>
</head>
<body>
    <h1>Cluster Report: {contrast_name}</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Threshold:</strong> {threshold} (p &lt; {1-threshold:.3f})</p>
        <p><strong>Number of clusters:</strong> {len(df)}</p>
        <p><strong>Total significant voxels:</strong> {df['Voxels'].sum() if 'Voxels' in df.columns else 0}</p>
    </div>

    <h2>Clusters</h2>
    {df.to_html(index=False, border=0)}

</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html)


def generate_reports_for_all_contrasts(
    randomise_output_dir: Path,
    output_dir: Path,
    contrast_names: Optional[List[str]] = None,
    threshold: float = 0.95,
    min_cluster_size: int = 10
) -> Dict:
    """
    Generate cluster reports for all contrasts in randomise output

    Args:
        randomise_output_dir: Directory containing randomise outputs
        output_dir: Directory for cluster reports
        contrast_names: Optional list of contrast names (order matches contrasts)
        threshold: Significance threshold
        min_cluster_size: Minimum cluster size

    Returns:
        Dictionary summarizing all reports
    """
    randomise_output_dir = Path(randomise_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all corrected p-value maps
    corrp_files = sorted(randomise_output_dir.glob('*_tfce_corrp_tstat*.nii.gz'))

    reports = []

    for i, corrp_file in enumerate(corrp_files):
        # Find corresponding stat file
        # corrp file: randomise_tfce_corrp_tstat1.nii.gz
        # stat file: randomise_tstat1.nii.gz
        stat_name = corrp_file.name.replace('_tfce_corrp_', '_').replace('_corrp', '')
        stat_file = randomise_output_dir / stat_name

        if not stat_file.exists():
            logging.warning(f"Stat file not found for {corrp_file}: {stat_file}")
            continue

        # Get contrast name
        if contrast_names and i < len(contrast_names):
            contrast_name = contrast_names[i]
        else:
            contrast_name = f"contrast_{i+1}"

        # Generate report
        report = generate_cluster_report(
            stat_file=stat_file,
            corrp_file=corrp_file,
            contrast_name=contrast_name,
            output_dir=output_dir,
            threshold=threshold,
            min_cluster_size=min_cluster_size
        )

        report['contrast_name'] = contrast_name
        reports.append(report)

    return {
        'reports': reports,
        'output_dir': str(output_dir)
    }


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Generate reports for all contrasts
    result = generate_reports_for_all_contrasts(
        randomise_output_dir=Path('/study/analysis/tbss_FA/model1/randomise_output/'),
        output_dir=Path('/study/analysis/tbss_FA/model1/cluster_reports/'),
        contrast_names=['age_positive', 'sex_MvsF', 'exposure_negative'],
        threshold=0.95,
        min_cluster_size=10
    )

    print(f"\nGenerated {len(result['reports'])} cluster reports")
    for report in result['reports']:
        if report['significant']:
            print(f"  {report['contrast_name']}: {report['n_clusters']} clusters, "
                  f"{report['total_voxels']} voxels")
        else:
            print(f"  {report['contrast_name']}: No significant clusters")
