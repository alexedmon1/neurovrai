#!/usr/bin/env python3
"""
Enhanced TBSS cluster reporting with atlas-based localization and visualization
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
import subprocess
import json

# JHU white matter atlas labels (FSL's JHU-ICBM-DTI-81)
JHU_LABELS = {
    0: "Background",
    1: "Middle cerebellar peduncle",
    2: "Pontine crossing tract",
    3: "Genu of corpus callosum",
    4: "Body of corpus callosum",
    5: "Splenium of corpus callosum",
    6: "Fornix (column and body of fornix)",
    7: "Corticospinal tract R",
    8: "Corticospinal tract L",
    9: "Medial lemniscus R",
    10: "Medial lemniscus L",
    11: "Inferior cerebellar peduncle R",
    12: "Inferior cerebellar peduncle L",
    13: "Superior cerebellar peduncle R",
    14: "Superior cerebellar peduncle L",
    15: "Cerebral peduncle R",
    16: "Cerebral peduncle L",
    17: "Anterior limb of internal capsule R",
    18: "Anterior limb of internal capsule L",
    19: "Posterior limb of internal capsule R",
    20: "Posterior limb of internal capsule L",
    21: "Retrolenticular part of internal capsule R",
    22: "Retrolenticular part of internal capsule L",
    23: "Anterior corona radiata R",
    24: "Anterior corona radiata L",
    25: "Superior corona radiata R",
    26: "Superior corona radiata L",
    27: "Posterior corona radiata R",
    28: "Posterior corona radiata L",
    29: "Posterior thalamic radiation R",
    30: "Posterior thalamic radiation L",
    31: "Sagittal stratum R",
    32: "Sagittal stratum L",
    33: "External capsule R",
    34: "External capsule L",
    35: "Cingulum (cingulate gyrus) R",
    36: "Cingulum (cingulate gyrus) L",
    37: "Cingulum (hippocampus) R",
    38: "Cingulum (hippocampus) L",
    39: "Fornix (cres) / Stria terminalis R",
    40: "Fornix (cres) / Stria terminalis L",
    41: "Superior longitudinal fasciculus R",
    42: "Superior longitudinal fasciculus L",
    43: "Superior fronto-occipital fasciculus R",
    44: "Superior fronto-occipital fasciculus L",
    45: "Uncinate fasciculus R",
    46: "Uncinate fasciculus L",
    47: "Tapetum R",
    48: "Tapetum L"
}

def load_jhu_atlas():
    """Load JHU white matter atlas from FSL"""
    fsl_dir = subprocess.check_output(['echo', '$FSLDIR'],
                                     shell=True).decode().strip()
    if not fsl_dir or fsl_dir == '$FSLDIR':
        fsl_dir = '/usr/local/fsl'

    atlas_path = Path(fsl_dir) / 'data' / 'atlases' / 'JHU' / 'JHU-ICBM-labels-1mm.nii.gz'

    if atlas_path.exists():
        return nib.load(str(atlas_path))
    else:
        print(f"Warning: JHU atlas not found at {atlas_path}")
        return None

def identify_cluster_location(cluster_mask, atlas_img, atlas_labels):
    """Identify anatomical location of cluster using atlas"""
    if atlas_img is None:
        return []

    atlas_data = atlas_img.get_fdata()

    # Get atlas labels that overlap with cluster
    cluster_coords = np.where(cluster_mask)
    overlapping_labels = []

    for x, y, z in zip(*cluster_coords):
        if (x < atlas_data.shape[0] and y < atlas_data.shape[1] and
            z < atlas_data.shape[2]):
            label_id = int(atlas_data[x, y, z])
            if label_id > 0:
                overlapping_labels.append(label_id)

    # Count frequency of each label
    if overlapping_labels:
        label_counts = {}
        for label_id in overlapping_labels:
            label_name = atlas_labels.get(label_id, f"Unknown-{label_id}")
            label_counts[label_name] = label_counts.get(label_name, 0) + 1

        # Sort by frequency
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

        # Calculate percentages
        total_voxels = len(overlapping_labels)
        locations = []
        for label_name, count in sorted_labels[:3]:  # Top 3 regions
            pct = (count / total_voxels) * 100
            locations.append({
                'region': label_name,
                'voxels': count,
                'percentage': pct
            })

        return locations

    return []

def create_mosaic_visualization(stat_data, cluster_mask, bg_data, output_file,
                                cluster_id, peak_coords):
    """Create mosaic view (axial, coronal, sagittal) of cluster"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Get peak coordinates
    peak_x, peak_y, peak_z = peak_coords

    # Axial view (Z slice)
    ax = axes[0]
    ax.imshow(bg_data[:, :, peak_z].T, cmap='gray', origin='lower', aspect='auto')
    ax.contour(cluster_mask[:, :, peak_z].T, colors='red', linewidths=2, levels=[0.5])
    ax.set_title(f'Axial (Z={peak_z})', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Coronal view (Y slice)
    ax = axes[1]
    ax.imshow(bg_data[:, peak_y, :].T, cmap='gray', origin='lower', aspect='auto')
    ax.contour(cluster_mask[:, peak_y, :].T, colors='red', linewidths=2, levels=[0.5])
    ax.set_title(f'Coronal (Y={peak_y})', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Sagittal view (X slice)
    ax = axes[2]
    ax.imshow(bg_data[peak_x, :, :].T, cmap='gray', origin='lower', aspect='auto')
    ax.contour(cluster_mask[peak_x, :, :].T, colors='red', linewidths=2, levels=[0.5])
    ax.set_title(f'Sagittal (X={peak_x})', fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.suptitle(f'Cluster {cluster_id} Localization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def extract_clusters_with_atlas(stat_file, corrp_file, mean_fa_file,
                                p_thresh=0.7, min_size=5, top_n=10):
    """Extract clusters with anatomical localization"""

    # Load images
    stat_img = nib.load(stat_file)
    corrp_img = nib.load(corrp_file)
    mean_fa_img = nib.load(mean_fa_file)

    stat_data = stat_img.get_fdata()
    corrp_data = corrp_img.get_fdata()
    mean_fa_data = mean_fa_img.get_fdata()

    # Load atlas
    atlas_img = load_jhu_atlas()

    # Threshold by p-value
    sig_mask = corrp_data >= p_thresh

    # Label connected components
    labeled, n_clusters = ndimage.label(sig_mask)

    clusters = []
    for i in range(1, n_clusters + 1):
        cluster_mask = labeled == i
        cluster_size = np.sum(cluster_mask)

        if cluster_size >= min_size:
            # Get cluster statistics
            cluster_stats = stat_data[cluster_mask]
            cluster_corrp = corrp_data[cluster_mask]

            # Get peak location
            peak_idx = np.argmax(cluster_stats)
            peak_coords = np.where(cluster_mask)
            peak_loc = (int(peak_coords[0][peak_idx]),
                       int(peak_coords[1][peak_idx]),
                       int(peak_coords[2][peak_idx]))

            # Identify anatomical location
            locations = identify_cluster_location(cluster_mask, atlas_img, JHU_LABELS)

            clusters.append({
                'cluster_id': i,
                'size': int(cluster_size),
                'peak_stat': float(np.max(cluster_stats)),
                'peak_p': float(1 - np.max(cluster_corrp)),
                'mean_stat': float(np.mean(cluster_stats)),
                'peak_coords': peak_loc,
                'peak_corrp': float(np.max(cluster_corrp)),
                'locations': locations,
                'mask': cluster_mask
            })

    # Sort by peak stat and take top N
    clusters_sorted = sorted(clusters, key=lambda x: x['peak_stat'], reverse=True)
    top_clusters = clusters_sorted[:top_n]

    return top_clusters, mean_fa_data

def generate_enhanced_html_report(clusters, mean_fa_data, contrast_name,
                                  output_dir, stat_file):
    """Generate enhanced HTML report with visualizations"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate mosaic images for top clusters
    img_dir = output_dir / 'images'
    img_dir.mkdir(exist_ok=True)

    for cluster in clusters:
        img_file = img_dir / f"cluster_{cluster['cluster_id']:03d}.png"
        create_mosaic_visualization(
            stat_data=nib.load(stat_file).get_fdata(),
            cluster_mask=cluster['mask'],
            bg_data=mean_fa_data,
            output_file=img_file,
            cluster_id=cluster['cluster_id'],
            peak_coords=cluster['peak_coords']
        )
        cluster['image_file'] = f"images/cluster_{cluster['cluster_id']:03d}.png"

    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced TBSS Cluster Report - {contrast_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
            font-size: 2.2em;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            font-size: 1.6em;
        }}
        .summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            margin: 25px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .summary-item {{
            display: inline-block;
            margin-right: 40px;
            font-weight: bold;
            font-size: 1.1em;
        }}
        .cluster-card {{
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 10px;
            padding: 25px;
            margin: 30px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .cluster-header {{
            background: linear-gradient(90deg, #3498db, #2980b9);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin: -25px -25px 20px -25px;
            font-size: 1.3em;
            font-weight: bold;
        }}
        .cluster-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .stat-value {{
            color: #2c3e50;
            font-size: 1.4em;
            font-weight: bold;
        }}
        .locations {{
            background: white;
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
        }}
        .location-item {{
            padding: 10px;
            margin: 5px 0;
            background: #ecf0f1;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .location-name {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .location-pct {{
            background: #3498db;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
        .mosaic-img {{
            width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            margin: 20px 0;
        }}
        .p-significant {{
            color: #27ae60;
            font-weight: bold;
        }}
        .p-marginal {{
            color: #f39c12;
            font-weight: bold;
        }}
        .p-ns {{
            color: #95a5a6;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 30px;
            border-top: 2px solid #dee2e6;
            color: #7f8c8d;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Enhanced TBSS Cluster Report</h1>
        <h2>Contrast: {contrast_name}</h2>

        <div class="summary">
            <div class="summary-item">📊 Total Clusters: {len(clusters)}</div>
            <div class="summary-item">🎯 Threshold: p &lt; 0.30</div>
            <div class="summary-item">📏 Min Size: 5 voxels</div>
            <div class="summary-item">🏆 Showing Top 10 Clusters</div>
        </div>
"""

    # Add cluster cards
    for rank, cluster in enumerate(clusters, 1):
        p_class = "p-significant" if cluster['peak_p'] < 0.05 else "p-marginal" if cluster['peak_p'] < 0.30 else "p-ns"

        html += f"""
        <div class="cluster-card">
            <div class="cluster-header">
                #{rank} - Cluster {cluster['cluster_id']}
            </div>

            <div class="cluster-stats">
                <div class="stat-box">
                    <div class="stat-label">Cluster Size</div>
                    <div class="stat-value">{cluster['size']} voxels</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Peak T-statistic</div>
                    <div class="stat-value">{cluster['peak_stat']:.3f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Peak p-value</div>
                    <div class="stat-value {p_class}">p = {cluster['peak_p']:.4f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Peak Location</div>
                    <div class="stat-value" style="font-size: 1.0em;">
                        ({cluster['peak_coords'][0]}, {cluster['peak_coords'][1]}, {cluster['peak_coords'][2]})
                    </div>
                </div>
            </div>

            <h3>📍 Anatomical Location (JHU White Matter Atlas)</h3>
            <div class="locations">
"""

        if cluster['locations']:
            for loc in cluster['locations']:
                html += f"""
                <div class="location-item">
                    <span class="location-name">{loc['region']}</span>
                    <span class="location-pct">{loc['percentage']:.1f}% ({loc['voxels']} voxels)</span>
                </div>
"""
        else:
            html += """
                <div class="location-item">
                    <span class="location-name" style="color: #95a5a6;">No atlas labels found</span>
                </div>
"""

        html += f"""
            </div>

            <h3>🔍 Visualization</h3>
            <img src="{cluster['image_file']}" class="mosaic-img" alt="Cluster {cluster['cluster_id']} visualization">
        </div>
"""

    html += """
        <div class="footer">
            <p><strong>Note:</strong> This report uses liberal thresholds (p &lt; 0.30) for demonstration.</p>
            <p>For publication-quality results, use standard thresholds (p &lt; 0.05 corrected) with ≥5000 permutations.</p>
            <p><strong>Atlas:</strong> JHU ICBM-DTI-81 White Matter Labels</p>
            <p>Generated by neurovrai enhanced TBSS analysis pipeline</p>
        </div>
    </div>
</body>
</html>
"""

    output_file = output_dir / f"{contrast_name}_enhanced_report.html"
    with open(output_file, 'w') as f:
        f.write(html)

    return output_file

# Main execution
if __name__ == '__main__':
    import sys

    stat_file = sys.argv[1]
    corrp_file = sys.argv[2]
    mean_fa_file = sys.argv[3]
    contrast_name = sys.argv[4]
    output_dir = sys.argv[5]
    p_thresh = float(sys.argv[6]) if len(sys.argv) > 6 else 0.7

    print(f"\n{'='*80}")
    print(f"Enhanced TBSS Cluster Report Generation")
    print(f"{'='*80}")
    print(f"Contrast: {contrast_name}")
    print(f"Threshold: p < {1-p_thresh:.2f} (corrp >= {p_thresh})")
    print(f"{'='*80}\n")

    # Extract clusters with atlas localization
    print("📊 Extracting clusters and identifying anatomical locations...")
    clusters, mean_fa_data = extract_clusters_with_atlas(
        stat_file=stat_file,
        corrp_file=corrp_file,
        mean_fa_file=mean_fa_file,
        p_thresh=p_thresh,
        min_size=5,
        top_n=10
    )

    print(f"   Found {len(clusters)} top clusters\n")

    # Generate visualizations and HTML report
    print("🎨 Generating mosaic visualizations...")
    output_file = generate_enhanced_html_report(
        clusters=clusters,
        mean_fa_data=mean_fa_data,
        contrast_name=contrast_name,
        output_dir=output_dir,
        stat_file=stat_file
    )

    print(f"   ✅ Report saved: {output_file}\n")

    # Print summary
    print(f"{'='*80}")
    print("Top 10 Clusters Summary:")
    print(f"{'='*80}")
    for rank, c in enumerate(clusters, 1):
        print(f"\n#{rank} - Cluster {c['cluster_id']}")
        print(f"   Size: {c['size']} voxels")
        print(f"   Peak T: {c['peak_stat']:.3f}, p = {c['peak_p']:.4f}")
        if c['locations']:
            print(f"   Location: {c['locations'][0]['region']} ({c['locations'][0]['percentage']:.1f}%)")
        else:
            print(f"   Location: Unknown")
    print(f"\n{'='*80}\n")
