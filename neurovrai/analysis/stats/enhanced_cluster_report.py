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

# JHU white matter atlas labels (FSL's JHU-ICBM-DTI-81) - 48 regions
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

# JHU White Matter Tractography Atlas - 20 major tracts (more comprehensive)
JHU_TRACTS_LABELS = {
    0: "Background",
    1: "Anterior thalamic radiation L",
    2: "Anterior thalamic radiation R",
    3: "Corticospinal tract L",
    4: "Corticospinal tract R",
    5: "Cingulum (cingulate gyrus) L",
    6: "Cingulum (cingulate gyrus) R",
    7: "Cingulum (hippocampus) L",
    8: "Cingulum (hippocampus) R",
    9: "Forceps major (splenium of corpus callosum)",
    10: "Forceps minor (genu of corpus callosum)",
    11: "Inferior fronto-occipital fasciculus L",
    12: "Inferior fronto-occipital fasciculus R",
    13: "Inferior longitudinal fasciculus L",
    14: "Inferior longitudinal fasciculus R",
    15: "Superior longitudinal fasciculus L",
    16: "Superior longitudinal fasciculus R",
    17: "Uncinate fasciculus L",
    18: "Uncinate fasciculus R",
    19: "Superior longitudinal fasciculus (temporal part) L",
    20: "Superior longitudinal fasciculus (temporal part) R"
}

# Harvard-Oxford Cortical atlas labels (48 regions)
HARVARD_OXFORD_CORTICAL = {
    0: "Frontal Pole",
    1: "Insular Cortex",
    2: "Superior Frontal Gyrus",
    3: "Middle Frontal Gyrus",
    4: "Inferior Frontal Gyrus, pars triangularis",
    5: "Inferior Frontal Gyrus, pars opercularis",
    6: "Precentral Gyrus",
    7: "Temporal Pole",
    8: "Superior Temporal Gyrus, anterior division",
    9: "Superior Temporal Gyrus, posterior division",
    10: "Middle Temporal Gyrus, anterior division",
    11: "Middle Temporal Gyrus, posterior division",
    12: "Middle Temporal Gyrus, temporooccipital part",
    13: "Inferior Temporal Gyrus, anterior division",
    14: "Inferior Temporal Gyrus, posterior division",
    15: "Inferior Temporal Gyrus, temporooccipital part",
    16: "Postcentral Gyrus",
    17: "Superior Parietal Lobule",
    18: "Supramarginal Gyrus, anterior division",
    19: "Supramarginal Gyrus, posterior division",
    20: "Angular Gyrus",
    21: "Lateral Occipital Cortex, superior division",
    22: "Lateral Occipital Cortex, inferior division",
    23: "Intracalcarine Cortex",
    24: "Frontal Medial Cortex",
    25: "Juxtapositional Lobule Cortex",
    26: "Subcallosal Cortex",
    27: "Paracingulate Gyrus",
    28: "Cingulate Gyrus, anterior division",
    29: "Cingulate Gyrus, posterior division",
    30: "Precuneous Cortex",
    31: "Cuneal Cortex",
    32: "Frontal Orbital Cortex",
    33: "Parahippocampal Gyrus, anterior division",
    34: "Parahippocampal Gyrus, posterior division",
    35: "Lingual Gyrus",
    36: "Temporal Fusiform Cortex, anterior division",
    37: "Temporal Fusiform Cortex, posterior division",
    38: "Temporal Occipital Fusiform Cortex",
    39: "Occipital Fusiform Gyrus",
    40: "Frontal Operculum Cortex",
    41: "Central Opercular Cortex",
    42: "Parietal Operculum Cortex",
    43: "Planum Polare",
    44: "Heschl's Gyrus",
    45: "Planum Temporale",
    46: "Supracalcarine Cortex",
    47: "Occipital Pole"
}

# Harvard-Oxford Subcortical atlas labels (21 regions)
HARVARD_OXFORD_SUBCORTICAL = {
    0: "Left Cerebral White Matter",
    1: "Left Cerebral Cortex",
    2: "Left Lateral Ventricle",
    3: "Left Thalamus",
    4: "Left Caudate",
    5: "Left Putamen",
    6: "Left Pallidum",
    7: "Brain-Stem",
    8: "Left Hippocampus",
    9: "Left Amygdala",
    10: "Left Accumbens",
    11: "Right Cerebral White Matter",
    12: "Right Cerebral Cortex",
    13: "Right Lateral Ventricle",
    14: "Right Thalamus",
    15: "Right Caudate",
    16: "Right Putamen",
    17: "Right Pallidum",
    18: "Right Hippocampus",
    19: "Right Amygdala",
    20: "Right Accumbens"
}

def get_fsl_dir():
    """Get FSL directory"""
    import os
    fsl_dir = os.getenv('FSLDIR', '/usr/local/fsl')
    return Path(fsl_dir)


def load_jhu_atlas(use_combined=True):
    """
    Load JHU white matter atlas from FSL.

    Parameters
    ----------
    use_combined : bool or str
        - True or 'combined': Combine both tractography and labels atlases (best coverage)
        - 'tracts': Use only JHU tractography atlas (20 major tracts)
        - False or 'labels': Use only JHU labels atlas (48 regions)

    Returns
    -------
    tuple
        (atlas_img, atlas_labels_dict)
    """
    fsl_dir = get_fsl_dir()

    # Paths
    tracts_path = fsl_dir / 'data' / 'atlases' / 'JHU' / 'JHU-ICBM-tracts-maxprob-thr0-1mm.nii.gz'
    labels_path = fsl_dir / 'data' / 'atlases' / 'JHU' / 'JHU-ICBM-labels-1mm.nii.gz'

    # Determine mode
    if use_combined is True or use_combined == 'combined':
        # COMBINED MODE: Best anatomical coverage
        if not tracts_path.exists() or not labels_path.exists():
            print(f"Warning: Cannot combine atlases - one or both not found")
            print(f"   Tracts: {tracts_path.exists()}")
            print(f"   Labels: {labels_path.exists()}")
            # Fallback to whichever is available
            if tracts_path.exists():
                return load_jhu_atlas(use_combined='tracts')
            elif labels_path.exists():
                return load_jhu_atlas(use_combined='labels')
            return None, None

        print(f"   Using Combined JHU Atlas (Tracts + Labels)")

        # Load both atlases
        tracts_img = nib.load(str(tracts_path))
        labels_img = nib.load(str(labels_path))

        tracts_data = tracts_img.get_fdata()
        labels_data = labels_img.get_fdata()

        # Combine: tracts take priority (they're major fiber bundles)
        # Labels fill in gaps with offset of 100 to avoid conflicts
        combined_data = np.zeros_like(tracts_data)
        combined_data = tracts_data.copy()  # Start with tracts

        # Add labels where tracts are empty (background)
        mask = tracts_data == 0
        combined_data[mask] = labels_data[mask] + 100  # Offset by 100

        # Create combined image
        combined_img = nib.Nifti1Image(combined_data, tracts_img.affine, tracts_img.header)

        # Create combined labels dictionary
        combined_labels = JHU_TRACTS_LABELS.copy()
        for idx, label in JHU_LABELS.items():
            if idx > 0:  # Skip background
                combined_labels[idx + 100] = label  # Offset labels

        return combined_img, combined_labels

    elif use_combined == 'tracts':
        # TRACTS ONLY MODE
        if tracts_path.exists():
            print(f"   Using JHU White Matter Tractography Atlas (20 tracts)")
            return nib.load(str(tracts_path)), JHU_TRACTS_LABELS
        else:
            print(f"Warning: JHU tracts atlas not found at {tracts_path}")
            return None, None

    else:  # 'labels' or False
        # LABELS ONLY MODE
        if labels_path.exists():
            print(f"   Using JHU White Matter Labels Atlas (48 regions)")
            return nib.load(str(labels_path)), JHU_LABELS
        else:
            print(f"Warning: JHU labels atlas not found at {labels_path}")
            return None, None


def load_harvard_oxford_atlas(resolution='2mm'):
    """
    Load Harvard-Oxford cortical and subcortical atlases from FSL.

    Parameters
    ----------
    resolution : str
        Resolution of atlas ('1mm' or '2mm', default '2mm')

    Returns
    -------
    tuple
        (combined_atlas_img, combined_labels_dict)
    """
    fsl_dir = get_fsl_dir()

    # Load cortical atlas (48 regions)
    cort_path = fsl_dir / 'data' / 'atlases' / 'HarvardOxford' / f'HarvardOxford-cort-maxprob-thr0-{resolution}.nii.gz'
    # Load subcortical atlas (21 regions)
    sub_path = fsl_dir / 'data' / 'atlases' / 'HarvardOxford' / f'HarvardOxford-sub-maxprob-thr0-{resolution}.nii.gz'

    if not cort_path.exists() or not sub_path.exists():
        print(f"Warning: Harvard-Oxford atlases not found")
        return None, None

    cort_img = nib.load(str(cort_path))
    sub_img = nib.load(str(sub_path))

    cort_data = cort_img.get_fdata()
    sub_data = sub_img.get_fdata()

    # Combine atlases: use subcortical where available, otherwise cortical
    # Offset cortical labels by 100 to avoid conflicts
    combined_data = np.zeros_like(cort_data)
    combined_data = sub_data.copy()  # Subcortical takes priority (deep structures)
    mask = sub_data == 0  # Where subcortical is empty
    combined_data[mask] = cort_data[mask] + 100  # Offset cortical labels

    # Create combined image
    combined_img = nib.Nifti1Image(combined_data, cort_img.affine, cort_img.header)

    # Create combined labels dictionary
    combined_labels = HARVARD_OXFORD_SUBCORTICAL.copy()
    for idx, label in HARVARD_OXFORD_CORTICAL.items():
        combined_labels[idx + 100] = label

    return combined_img, combined_labels

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
                                p_thresh=0.7, min_size=5, top_n=10,
                                atlas_type='jhu', atlas_resolution='2mm'):
    """
    Extract clusters with anatomical localization

    Parameters
    ----------
    stat_file : Path
        Statistical map file
    corrp_file : Path
        Corrected p-value map file
    mean_fa_file : Path
        Background image (FA for TBSS, T1 for VBM)
    p_thresh : float
        Corrected p-value threshold (corrp >= p_thresh)
    min_size : int
        Minimum cluster size in voxels
    top_n : int
        Number of top clusters to return
    atlas_type : str
        Atlas to use: 'jhu' for white matter (TBSS), 'harvard-oxford' for grey matter (VBM)
    atlas_resolution : str
        Atlas resolution ('1mm' or '2mm')

    Returns
    -------
    tuple
        (clusters, bg_data)
    """

    # Load images
    stat_img = nib.load(stat_file)
    corrp_img = nib.load(corrp_file)
    mean_fa_img = nib.load(mean_fa_file)

    stat_data = stat_img.get_fdata()
    corrp_data = corrp_img.get_fdata()
    mean_fa_data = mean_fa_img.get_fdata()

    # Load atlas based on type
    if atlas_type == 'harvard-oxford':
        atlas_img, atlas_labels = load_harvard_oxford_atlas(resolution=atlas_resolution)
    else:  # Default to JHU for TBSS (use combined atlas - best coverage)
        atlas_img, atlas_labels = load_jhu_atlas(use_combined=True)

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
            locations = identify_cluster_location(cluster_mask, atlas_img, atlas_labels)

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
                                  output_dir, stat_file, atlas_type='jhu',
                                  analysis_type=None, study_name=None,
                                  threshold_p=0.05):
    """Generate enhanced HTML report with visualizations

    Parameters
    ----------
    atlas_type : str
        Atlas used for localization: 'jhu' or 'harvard-oxford'
    analysis_type : str, optional
        Type of analysis (e.g., 'ReHo', 'fALFF', 'VBM', 'TBSS')
    study_name : str, optional
        Name of the study (e.g., 'mock_study')
    threshold_p : float
        P-value threshold used (default: 0.05)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set report title and atlas name based on atlas type or analysis_type
    if analysis_type:
        report_title = f"{analysis_type} Cluster Analysis Report"
    elif atlas_type == 'harvard-oxford':
        report_title = "VBM Cluster Analysis Report"
    else:
        report_title = "TBSS Cluster Analysis Report"

    if atlas_type == 'harvard-oxford':
        atlas_name = "Harvard-Oxford Cortical & Subcortical Atlas"
    else:
        atlas_name = "JHU Combined White Matter Atlas (Tractography + Labels)"

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

    # Get current date/time
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Study name line (if provided)
    study_line = f"<h3 style='color: #7f8c8d; margin-top: -10px;'>Study: {study_name}</h3>" if study_name else ""

    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_title} - {contrast_name}</title>
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
        h3 {{
            color: #7f8c8d;
            margin-top: 10px;
            font-size: 1.1em;
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
        <h1>🧠 {report_title}</h1>
        {study_line}
        <h3 style='color: #7f8c8d; margin-top: -10px;'>Generated: {timestamp}</h3>
        <h2>Contrast: {contrast_name}</h2>

        <div class="summary">
            <div class="summary-item">📊 Total Clusters: {len(clusters)}</div>
            <div class="summary-item">🎯 Threshold: p &lt; {threshold_p:.2f}</div>
            <div class="summary-item">📏 Min Size: 5 voxels</div>
            <div class="summary-item">🏆 Showing Top 10 Clusters</div>
            <div class="summary-item">🗺️ Atlas: {atlas_name}</div>
        </div>
"""

    # Add demographics section if provided
    if demographics:
        g1 = demographics['group1']
        g2 = demographics['group2']
        html += f"""
        <div class="summary" style="margin-top: 20px;">
            <div class="summary-item" style="grid-column: span 5;">
                <strong>📋 Sample Demographics (N={demographics['total_n']})</strong>
            </div>
            <div class="summary-item">
                <strong>{g1['label']}</strong><br>
                N={g1['n']} ({g1['n_male']}M/{g1['n_female']}F)<br>
                Age: M={g1['age_mean']:.1f}, SD={g1['age_std']:.1f}
            </div>
            <div class="summary-item">
                <strong>{g2['label']}</strong><br>
                N={g2['n']} ({g2['n_male']}M/{g2['n_female']}F)<br>
                Age: M={g2['age_mean']:.1f}, SD={g2['age_std']:.1f}
            </div>
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

            <h3>📍 Anatomical Location ({atlas_name})</h3>
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

    # Footer with dynamic threshold and atlas
    threshold_note = f"liberal thresholds (p &lt; {threshold_p:.2f})" if threshold_p > 0.05 else f"standard threshold (p &lt; {threshold_p:.2f})"

    html += f"""
        <div class="footer">
            <p><strong>Note:</strong> This report uses {threshold_note} for cluster detection.</p>
            <p>For publication-quality results, use standard thresholds (p &lt; 0.05 corrected) with ≥5000 permutations.</p>
            <p><strong>Atlas:</strong> {atlas_name}</p>
            <p>Generated by neurovrai enhanced cluster analysis pipeline</p>
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


def create_enhanced_cluster_report(stat_map, corrp_map, threshold, output_dir,
                                  contrast_name, max_clusters=10, liberal_threshold=0.7,
                                  background_image=None, atlas_type='harvard-oxford',
                                  study_name=None, analysis_type=None, demographics=None):
    """
    Create enhanced cluster report for VBM analysis.

    Wrapper function that adapts TBSS cluster reporting for VBM use.

    Parameters
    ----------
    stat_map : Path
        Path to tstat map (randomise_tstat*.nii.gz)
    corrp_map : Path
        Path to corrected p-value map (randomise_tfce_corrp_tstat*.nii.gz)
    threshold : float
        P-value threshold (e.g., 0.05)
    output_dir : Path
        Directory for output files
    contrast_name : str
        Name of contrast
    max_clusters : int
        Maximum number of clusters to report (default: 10)
    liberal_threshold : float
        Liberal p-value threshold for exploratory view (default: 0.7)
    background_image : Path, optional
        Background image for visualization (uses MNI152 if None)
    atlas_type : str
        Atlas to use: 'harvard-oxford' for grey matter (VBM, default), 'jhu' for white matter (TBSS)
    study_name : str, optional
        Name of the study (e.g., 'mock_study')
    analysis_type : str, optional
        Type of analysis (e.g., 'ReHo', 'fALFF', 'VBM', 'TBSS')

    Returns
    -------
    dict
        Dictionary with cluster results and report paths
    """
    from pathlib import Path
    import os

    stat_map = Path(stat_map)
    corrp_map = Path(corrp_map)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use MNI152 T1 2mm as background if not provided
    if background_image is None:
        fsldir = os.getenv('FSLDIR', '/usr/local/fsl')
        background_image = Path(fsldir) / 'data' / 'standard' / 'MNI152_T1_2mm.nii.gz'
    else:
        background_image = Path(background_image)

    # Convert threshold to corrp threshold (e.g., p<0.05 -> corrp>0.95)
    corrp_threshold = 1.0 - threshold

    # Extract clusters (using mean FA slot for background image)
    try:
        clusters, bg_data = extract_clusters_with_atlas(
            stat_file=stat_map,
            corrp_file=corrp_map,
            mean_fa_file=background_image,
            p_thresh=corrp_threshold,
            min_size=5,
            top_n=max_clusters,
            atlas_type=atlas_type
        )
    except Exception as e:
        print(f"Warning: Could not extract clusters with atlas: {e}")
        print(f"Falling back to simple cluster extraction...")
        # Fallback: simple cluster extraction without atlas
        clusters = extract_simple_clusters(stat_map, corrp_map, corrp_threshold, max_clusters)
        bg_data = nib.load(background_image).get_fdata()

    # Determine analysis type from contrast_name if not already specified
    if analysis_type is None:
        # Try to extract analysis type from contrast name
        for atype in ['ReHo', 'fALFF', 'VBM', 'TBSS']:
            if atype.lower() in contrast_name.lower():
                analysis_type = atype
                break

    # Generate HTML report
    if clusters:
        report_html = generate_enhanced_html_report(
            clusters=clusters,
            mean_fa_data=bg_data,
            contrast_name=contrast_name,
            output_dir=output_dir,
            stat_file=stat_map,
            atlas_type=atlas_type,
            analysis_type=analysis_type,
            study_name=study_name,
            threshold_p=threshold
        )
    else:
        report_html = output_dir / f"{contrast_name}_report.html"
        with open(report_html, 'w') as f:
            f.write(f"<html><body><h1>No significant clusters found for {contrast_name}</h1>")
            f.write(f"<p>Threshold: p < {threshold}</p></body></html>")

    return {
        'n_clusters': len(clusters),
        'clusters': clusters,
        'report_html': str(report_html),
        'contrast_name': contrast_name,
        'threshold': threshold
    }


def extract_simple_clusters(stat_file, corrp_file, corrp_threshold, max_clusters=10):
    """
    Simple cluster extraction without atlas (fallback function).

    Parameters
    ----------
    stat_file : Path
        Path to tstat map
    corrp_file : Path
        Path to corrected p-value map
    corrp_threshold : float
        Corrected p-value threshold (e.g., 0.95 for p<0.05)
    max_clusters : int
        Maximum number of clusters

    Returns
    -------
    list
        List of cluster dictionaries
    """
    import nibabel as nib
    import numpy as np
    from scipy import ndimage

    # Load images
    stat_img = nib.load(stat_file)
    corrp_img = nib.load(corrp_file)

    stat_data = stat_img.get_fdata()
    corrp_data = corrp_img.get_fdata()

    # Threshold
    sig_mask = corrp_data >= corrp_threshold

    # Label connected components
    labeled, n_clusters = ndimage.label(sig_mask)

    clusters = []
    for i in range(1, n_clusters + 1):
        cluster_mask = labeled == i
        cluster_size = np.sum(cluster_mask)

        if cluster_size < 5:  # Minimum cluster size
            continue

        # Get stats
        cluster_stats = stat_data[cluster_mask]
        cluster_corrp = corrp_data[cluster_mask]

        # Peak location
        peak_idx = np.argmax(cluster_corrp)
        peak_loc = tuple(np.array(np.where(cluster_mask)).T[peak_idx])

        clusters.append({
            'cluster_id': i,
            'size': int(cluster_size),
            'peak_stat': float(np.max(cluster_stats)),
            'peak_p': float(1 - np.max(cluster_corrp)),
            'mean_stat': float(np.mean(cluster_stats)),
            'peak_coords': peak_loc,
            'peak_corrp': float(np.max(cluster_corrp)),
            'locations': [],  # No atlas info in fallback
            'mask': cluster_mask
        })

    # Sort by peak stat and return top N
    clusters_sorted = sorted(clusters, key=lambda x: x['peak_stat'], reverse=True)
    return clusters_sorted[:max_clusters]
