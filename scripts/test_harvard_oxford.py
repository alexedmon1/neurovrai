#!/usr/bin/env python3
"""Test Harvard-Oxford atlas for VBM cluster reporting"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurovrai.analysis.stats.enhanced_cluster_report import create_enhanced_cluster_report

# Paths
stats_dir = Path('/mnt/bytopia/IRC805/analysis/vbm/GM/stats')
randomise_dir = stats_dir / 'randomise_output'
output_dir = stats_dir / 'cluster_reports_harvard_oxford'
output_dir.mkdir(exist_ok=True)

# Test with age_negative at p<0.30 (corrp>0.70)
contrast_name = 'age_negative'
stat_map = randomise_dir / 'randomise_tstat2.nii.gz'
corrp_map = randomise_dir / 'randomise_tfce_corrp_tstat2.nii.gz'

print(f"Testing Harvard-Oxford atlas for {contrast_name}...")
print(f"Stat map: {stat_map}")
print(f"Corrp map: {corrp_map}")
print(f"Output dir: {output_dir}")

results = create_enhanced_cluster_report(
    stat_map=stat_map,
    corrp_map=corrp_map,
    threshold=0.30,
    output_dir=output_dir,
    contrast_name=f'{contrast_name}_p030',
    max_clusters=10,
    atlas_type='harvard-oxford'
)

print(f"\nResults:")
print(f"  Number of clusters: {results['n_clusters']}")
print(f"  Report HTML: {results['report_html']}")
print(f"\nCluster details:")
for i, cluster in enumerate(results['clusters'][:5], 1):
    print(f"\nCluster {i}:")
    print(f"  Size: {cluster['size']} voxels")
    print(f"  Peak p: {cluster['peak_p']:.4f}")
    print(f"  Peak stat: {cluster['peak_stat']:.2f}")
    if cluster['locations']:
        print(f"  Locations: {cluster['locations'][:3]}")
    else:
        print(f"  Locations: None found")

