# Neuroimaging Methods (Brief Version)

## Methods Section (250 words)

Neuroimaging data will be processed using our comprehensive neurovrai pipeline (Edmondson, 2025) integrating FSL (Jenkinson et al., 2012), ANTs (Tustison et al., 2024), and FreeSurfer 7.4 (Fischl et al., 2002). Structural T1-weighted images will undergo skull stripping (BET), bias field correction (ANTs N4), and tissue segmentation (ANTs Atropos). FreeSurfer will provide automated cortical reconstruction, subcortical segmentation, and hippocampal subfield analysis, generating cortical thickness, surface area, and volumetric measures across regions defined by the Desikan-Killiany (68 regions) and Destrieux (148 regions) atlases.

Multi-shell diffusion MRI will be preprocessed using FSL's TOPUP for distortion correction (Andersson et al., 2003) and eddy for motion/eddy current correction (Andersson & Sotiropoulos, 2016). We will compute diffusion tensor metrics (FA, MD, RD, AD), diffusion kurtosis indices, and NODDI parameters (neurite density, orientation dispersion, isotropic fraction) using AMICO acceleration (Daducci et al., 2015), providing 100-fold speed improvement. Tract-based spatial statistics (TBSS; Smith et al., 2006) will enable voxelwise analysis of white matter microstructure across groups.

Multi-echo resting-state fMRI will undergo slice-timing correction, motion correction (MCFLIRT), and denoising using TEDANA for optimal echo combination and ICA-based artifact removal (DuPre et al., 2021). Regional homogeneity (Zang et al., 2004), fractional amplitude of low-frequency fluctuations (Zou et al., 2008), and group ICA via MELODIC will assess local activity and resting-state networks. Functional connectivity matrices will be constructed using FreeSurfer-derived parcellations.

Arterial spin labeling (ASL) will quantify cerebral blood flow using automated calibration and partial volume correction (Chappell et al., 2011). White matter hyperintensities will be automatically segmented from FLAIR/T2 images (Schmidt et al., 2012). Statistical analyses will employ FSL's randomise with threshold-free cluster enhancement for multiple comparison correction (Winkler et al., 2014).

## References (APA Format)

Andersson, J. L. R., Skare, S., & Ashburner, J. (2003). How to correct susceptibility distortions in spin-echo echo-planar images: Application to diffusion tensor imaging. *NeuroImage*, *20*(2), 870-888. https://doi.org/10.1016/S1053-8119(03)00336-7

Andersson, J. L. R., & Sotiropoulos, S. N. (2016). An integrated approach to correction for off-resonance effects and subject movement in diffusion MR imaging. *NeuroImage*, *125*, 1063-1078. https://doi.org/10.1016/j.neuroimage.2015.10.019

Chappell, M. A., MacIntosh, B. J., Donahue, M. J., Jezzard, P., & Woolrich, M. W. (2011). Partial volume correction of multiple inversion time arterial spin labeling MRI data. *Magnetic Resonance in Medicine*, *65*(4), 1173-1183. https://doi.org/10.1002/mrm.22641

Daducci, A., Canales-Rodríguez, E. J., Zhang, H., Dyrby, T. B., Alexander, D. C., & Thiran, J. P. (2015). Accelerated Microstructure Imaging via Convex Optimization (AMICO) from diffusion MRI data. *NeuroImage*, *105*, 32-44. https://doi.org/10.1016/j.neuroimage.2014.10.026

DuPre, E., Salo, T., Ahmed, Z., Bandettini, P. A., Bottenhorn, K. L., Caballero-Gaudes, C., Dowdle, L. T., Gonzalez-Castillo, J., Heunis, S., Kundu, P., Laird, A. R., Markello, R., Markiewicz, C. J., Moia, S., Staden, I., Teves, J. B., Uruñuela, E., Vaziri-Pashkam, M., Whitaker, K., & Handwerker, D. A. (2021). TE-dependent analysis of multi-echo fMRI with tedana. *Journal of Open Source Software*, *6*(66), 3669. https://doi.org/10.21105/joss.03669

Edmondson, A. (2025). *neurovrai: Comprehensive MRI preprocessing and analysis platform* (Version 0.2.0) [Computer software]. https://github.com/alexedmon1/neurovrai

Fischl, B., Salat, D. H., Busa, E., Albert, M., Dieterich, M., Haselgrove, C., van der Kouwe, A., Killiany, R., Kennedy, D., Klaveness, S., Montillo, A., Makris, N., Rosen, B., & Dale, A. M. (2002). Whole brain segmentation: Automated labeling of neuroanatomical structures in the human brain. *Neuron*, *33*(3), 341-355. https://doi.org/10.1016/S0896-6273(02)00569-X

Jenkinson, M., Beckmann, C. F., Behrens, T. E., Woolrich, M. W., & Smith, S. M. (2012). FSL. *NeuroImage*, *62*(2), 782-790. https://doi.org/10.1016/j.neuroimage.2011.09.015

Schmidt, P., Gaser, C., Arsic, M., Buck, D., Förschler, A., Berthele, A., Hoshi, M., Ilg, R., Schmid, V. J., Zimmer, C., Hemmer, B., & Mühlau, M. (2012). An automated tool for detection of FLAIR-hyperintense white-matter lesions in multiple sclerosis. *NeuroImage*, *59*(4), 3774-3783. https://doi.org/10.1016/j.neuroimage.2011.11.032

Smith, S. M., Jenkinson, M., Johansen-Berg, H., Rueckert, D., Nichols, T. E., Mackay, C. E., Watkins, K. E., Ciccarelli, O., Cader, M. Z., Matthews, P. M., & Behrens, T. E. J. (2006). Tract-based spatial statistics: Voxelwise analysis of multi-subject diffusion data. *NeuroImage*, *31*(4), 1487-1505. https://doi.org/10.1016/j.neuroimage.2006.02.024

Tustison, N. J., Yassa, M. A., Rizvi, B., Cook, P. A., Holbrook, A. J., Srinivasan, M. T., Duda, J. T., Mangan, J. M., Stone, J. R., Manzanera, E. C., Shah, D., Satterthwaite, T. D., Elliott, M. A., Cieslak, M. C., Sydnor, V. J., Davatzikos, C., Oathes, D. J., Dolui, S., Detre, J. A., ... Avants, B. B. (2024). ANTsX neuroimaging-derived structural phenotypes of UK Biobank. *Scientific Reports*, *14*, 8848. https://doi.org/10.1038/s41598-024-59440-6

Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014). Permutation inference for the general linear model. *NeuroImage*, *92*, 381-397. https://doi.org/10.1016/j.neuroimage.2014.01.060

Zang, Y., Jiang, T., Lu, Y., He, Y., & Tian, L. (2004). Regional homogeneity approach to fMRI data analysis. *NeuroImage*, *22*(1), 394-400. https://doi.org/10.1016/j.neuroimage.2003.12.030

Zou, Q. H., Zhu, C. Z., Yang, Y., Zuo, X. N., Long, X. Y., Cao, Q. J., Wang, Y. F., & Zang, Y. F. (2008). An improved approach to detection of amplitude of low-frequency fluctuation (ALFF) for resting-state fMRI: Fractional ALFF. *Journal of Neuroscience Methods*, *172*(1), 137-141. https://doi.org/10.1016/j.jneumeth.2008.04.012