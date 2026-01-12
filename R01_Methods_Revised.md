# Revised R01 Methods (More Authentic Version)

## Neuroimaging Acquisition and Preprocessing

Neuroimaging data will be processed using our neurovrai pipeline (Edmondson, 2025), which integrates FSL 6.0.7 (Jenkinson et al., 2012), ANTs 2.5 (Tustison et al., 2024), and FreeSurfer 7.4. T1-weighted images undergo skull stripping (BET, f=0.5), N4 bias correction, and Atropos segmentation with k-means initialization. FreeSurfer generates cortical reconstructions and parcellations using the Desikan-Killiany atlas (68 regions).

Multi-shell diffusion MRI (b=1000,2000,3000 s/mm²) is preprocessed with FSL's TOPUP for susceptibility correction and eddy for motion/eddy current correction. We compute standard DTI metrics and fit NODDI models using AMICO (Daducci et al., 2015), reducing computation time from ~25 minutes to 30 seconds per dataset. Multi-echo fMRI (TE=14.5,29,43.5ms) undergoes TEDANA denoising (DuPre et al., 2021), which separates BOLD from non-BOLD components based on TE-dependence. Pulsed ASL (PCASL, 3.6s labeling duration) is processed using FSL's BASIL for CBF quantification.

## Quality Assessment

Quality assessment relies on automated metrics generated during preprocessing. For structural MRI, we verify tissue segmentation proportions fall within expected ranges (GM: 30-60%, WM: 25-55%, CSF: 5-35% of intracranial volume). FreeSurfer outputs undergo visual inspection for gross errors; participants with failed reconstructions are excluded but not manually corrected due to sample size constraints.

Motion assessment uses framewise displacement calculated from realignment parameters. While we target FD<0.5mm for functional scans, we will report actual distributions rather than enforce hard cutoffs, as recent work suggests motion censoring can introduce bias. For diffusion MRI, we retain volumes with FD<1.0mm and use eddy's outlier replacement rather than excluding subjects. ASL motion sensitivity will be addressed through weighted CBF estimation rather than exclusion.

## Statistical Analysis

Voxelwise statistics use FSL's randomise with 5,000 permutations and TFCE for cluster-free inference (p<0.05 FWE-corrected). To address the limitation that t-statistics conflate effect size with sample size, we will generate standardized effect size maps (Cohen's d) from all contrasts, converting t-statistics using d = t × √(1/n₁ + 1/n₂) for group comparisons. This provides interpretable effect magnitudes (d=0.2 small, 0.5 medium, 0.8 large) independent of sample size.

We acknowledge that multi-modal integration remains challenging; we will analyze each modality separately, then examine spatial overlap of significant findings rather than attempting formal multi-modal fusion. Given our sample size (anticipated n=120), we will report both thresholded (p<0.05 FWE) and unthresholded effect size maps, allowing readers to evaluate the full pattern of effects beyond binary significance cutoffs. This approach provides transparency about both statistical significance and practical importance of our findings.