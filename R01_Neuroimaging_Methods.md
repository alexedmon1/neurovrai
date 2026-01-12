# Neuroimaging Data Processing and Analysis Methods

## Methods Section

All neuroimaging data will be processed using our comprehensive neurovrai pipeline (Edmondson, 2025), which integrates validated tools from FSL (Jenkinson et al., 2012), ANTs (Tustison et al., 2024; Avants et al., 2011), and Nipype workflow management. Structural T1-weighted images will undergo skull stripping using FSL's BET, bias field correction with ANTs N4 (Tustison et al., 2010), and tissue segmentation using ANTs Atropos with k-means clustering.

Cortical reconstruction and subcortical segmentation will be performed using FreeSurfer 7.4 (Dale et al., 1999; Fischl et al., 2002, 2004), which provides automated parcellation of cortical and subcortical structures. FreeSurfer's surface-based morphometry pipeline will generate cortical thickness measurements, surface area, and volume estimates for each region defined by the Desikan-Killiany (68 cortical regions) and Destrieux (148 cortical regions) atlases (Desikan et al., 2006; Destrieux et al., 2010). The hippocampal subfield segmentation module will be employed to examine subregion-specific volumetric changes potentially associated with gestational diabetes (Iglesias et al., 2015). FreeSurfer-derived anatomical parcellations will serve as regions of interest for both functional and structural connectivity analyses, with optimized registration procedures ensuring accurate transformation between FreeSurfer space, native diffusion space, and MNI standard space using boundary-based registration (Greve & Fischl, 2009).

Diffusion-weighted imaging data will be preprocessed using FSL's TOPUP to correct for susceptibility-induced distortions (Andersson et al., 2003) and eddy for motion and eddy current correction (Andersson & Sotiropoulos, 2016). Diffusion tensor metrics (FA, MD, RD, AD) will be calculated using FSL's dtifit. For multi-shell acquisitions, we will additionally compute diffusion kurtosis metrics and NODDI parameters (intracellular volume fraction, orientation dispersion index, isotropic volume fraction) using AMICO acceleration (Daducci et al., 2015), which provides a 100-fold speed improvement over conventional fitting methods. Structural connectivity will be assessed using probabilistic tractography with anatomical constraints derived from FreeSurfer segmentations, including ventricle avoidance masks and white matter waypoints to improve biological plausibility of reconstructed pathways.

Resting-state fMRI preprocessing will adapt to acquisition parameters: multi-echo data will be denoised using TEDANA (DuPre et al., 2021; Kundu et al., 2017), while single-echo acquisitions will undergo ICA-AROMA for automated motion artifact removal (Pruim et al., 2015). Following standard preprocessing (slice-timing correction, motion correction with MCFLIRT, spatial smoothing), we will compute regional homogeneity (ReHo; Zang et al., 2004) and fractional amplitude of low-frequency fluctuations (fALFF; Zou et al., 2008) as measures of local brain activity. Group-level independent component analysis will be performed using FSL's MELODIC (Beckmann & Smith, 2004) to identify resting-state networks, with dual regression to derive subject-specific spatial maps and timeseries. Functional connectivity matrices will be constructed using FreeSurfer-derived cortical and subcortical parcellations, enabling network-based analyses of gestational diabetes effects on brain functional organization.

White matter hyperintensities (WMH) will be automatically segmented from FLAIR and T2-weighted images using a multimodal approach that integrates tissue probability maps and spatial priors (Schmidt et al., 2012). WMH volumes will be quantified both globally and regionally using the JHU white matter atlas. All processed data will undergo systematic quality control including registration accuracy assessment, motion quantification, and visual inspection of key outputs. Statistical analyses will employ FSL's randomise for nonparametric permutation testing (Winkler et al., 2014) with threshold-free cluster enhancement for multiple comparison correction, and FreeSurfer's linear mixed effects modeling for vertex-wise cortical thickness comparisons accounting for multiple comparisons using Monte Carlo simulation.

This integrated approach leverages production-tested workflows optimized for parallel processing, ensuring reproducible and efficient analysis of multimodal neuroimaging data in gestational diabetes research.

## References (APA Format)

Andersson, J. L. R., Skare, S., & Ashburner, J. (2003). How to correct susceptibility distortions in spin-echo echo-planar images: Application to diffusion tensor imaging. *NeuroImage*, *20*(2), 870-888. https://doi.org/10.1016/S1053-8119(03)00336-7

Andersson, J. L. R., & Sotiropoulos, S. N. (2016). An integrated approach to correction for off-resonance effects and subject movement in diffusion MR imaging. *NeuroImage*, *125*, 1063-1078. https://doi.org/10.1016/j.neuroimage.2015.10.019

Avants, B. B., Tustison, N. J., Song, G., Cook, P. A., Klein, A., & Gee, J. C. (2011). A reproducible evaluation of ANTs similarity metric performance in brain image registration. *NeuroImage*, *54*(3), 2033-2044. https://doi.org/10.1016/j.neuroimage.2010.09.025

Beckmann, C. F., & Smith, S. M. (2004). Probabilistic independent component analysis for functional magnetic resonance imaging. *IEEE Transactions on Medical Imaging*, *23*(2), 137-152. https://doi.org/10.1109/TMI.2003.822821

Daducci, A., Canales-Rodríguez, E. J., Zhang, H., Dyrby, T. B., Alexander, D. C., & Thiran, J. P. (2015). Accelerated Microstructure Imaging via Convex Optimization (AMICO) from diffusion MRI data. *NeuroImage*, *105*, 32-44. https://doi.org/10.1016/j.neuroimage.2014.10.026

Dale, A. M., Fischl, B., & Sereno, M. I. (1999). Cortical surface-based analysis: I. Segmentation and surface reconstruction. *NeuroImage*, *9*(2), 179-194. https://doi.org/10.1006/nimg.1998.0395

Desikan, R. S., Ségonne, F., Fischl, B., Quinn, B. T., Dickerson, B. C., Blacker, D., Buckner, R. L., Dale, A. M., Maguire, R. P., Hyman, B. T., Albert, M. S., & Killiany, R. J. (2006). An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest. *NeuroImage*, *31*(3), 968-980. https://doi.org/10.1016/j.neuroimage.2006.01.021

Destrieux, C., Fischl, B., Dale, A., & Halgren, E. (2010). Automatic parcellation of human cortical gyri and sulci using standard anatomical nomenclature. *NeuroImage*, *53*(1), 1-15. https://doi.org/10.1016/j.neuroimage.2010.06.010

DuPre, E., Salo, T., Ahmed, Z., Bandettini, P. A., Bottenhorn, K. L., Caballero-Gaudes, C., Dowdle, L. T., Gonzalez-Castillo, J., Heunis, S., Kundu, P., Laird, A. R., Markello, R., Markiewicz, C. J., Moia, S., Staden, I., Teves, J. B., Uruñuela, E., Vaziri-Pashkam, M., Whitaker, K., & Handwerker, D. A. (2021). TE-dependent analysis of multi-echo fMRI with tedana. *Journal of Open Source Software*, *6*(66), 3669. https://doi.org/10.21105/joss.03669

Edmondson, A. (2025). *neurovrai: Comprehensive MRI preprocessing and analysis platform* (Version 0.2.0) [Computer software]. https://github.com/alexedmon1/neurovrai

Fischl, B., Salat, D. H., Busa, E., Albert, M., Dieterich, M., Haselgrove, C., van der Kouwe, A., Killiany, R., Kennedy, D., Klaveness, S., Montillo, A., Makris, N., Rosen, B., & Dale, A. M. (2002). Whole brain segmentation: Automated labeling of neuroanatomical structures in the human brain. *Neuron*, *33*(3), 341-355. https://doi.org/10.1016/S0896-6273(02)00569-X

Fischl, B., van der Kouwe, A., Destrieux, C., Halgren, E., Ségonne, F., Salat, D. H., Busa, E., Seidman, L. J., Goldstein, J., Kennedy, D., Caviness, V., Makris, N., Rosen, B., & Dale, A. M. (2004). Automatically parcellating the human cerebral cortex. *Cerebral Cortex*, *14*(1), 11-22. https://doi.org/10.1093/cercor/bhg087

Greve, D. N., & Fischl, B. (2009). Accurate and robust brain image alignment using boundary-based registration. *NeuroImage*, *48*(1), 63-72. https://doi.org/10.1016/j.neuroimage.2009.06.060

Iglesias, J. E., Augustinack, J. C., Nguyen, K., Player, C. M., Player, A., Wright, M., Roy, N., Frosch, M. P., McKee, A. C., Wald, L. L., Fischl, B., & Van Leemput, K. (2015). A computational atlas of the hippocampal formation using ex vivo, ultra-high resolution MRI: Application to adaptive segmentation of in vivo MRI. *NeuroImage*, *115*, 117-137. https://doi.org/10.1016/j.neuroimage.2015.04.042

Jenkinson, M., Beckmann, C. F., Behrens, T. E., Woolrich, M. W., & Smith, S. M. (2012). FSL. *NeuroImage*, *62*(2), 782-790. https://doi.org/10.1016/j.neuroimage.2011.09.015

Kundu, P., Voon, V., Balchandani, P., Lombardo, M. V., Poser, B. A., & Bandettini, P. A. (2017). Multi-echo fMRI: A review of applications in fMRI denoising and analysis of BOLD signals. *NeuroImage*, *154*, 59-80. https://doi.org/10.1016/j.neuroimage.2017.03.033

Pruim, R. H. R., Mennes, M., van Rooij, D., Llera, A., Buitelaar, J. K., & Beckmann, C. F. (2015). ICA-AROMA: A robust ICA-based strategy for removing motion artifacts from fMRI data. *NeuroImage*, *112*, 267-277. https://doi.org/10.1016/j.neuroimage.2015.02.064

Schmidt, P., Gaser, C., Arsic, M., Buck, D., Förschler, A., Berthele, A., Hoshi, M., Ilg, R., Schmid, V. J., Zimmer, C., Hemmer, B., & Mühlau, M. (2012). An automated tool for detection of FLAIR-hyperintense white-matter lesions in multiple sclerosis. *NeuroImage*, *59*(4), 3774-3783. https://doi.org/10.1016/j.neuroimage.2011.11.032

Tustison, N. J., Avants, B. B., Cook, P. A., Zheng, Y., Egan, A., Yushkevich, P. A., & Gee, J. C. (2010). N4ITK: Improved N3 bias correction. *IEEE Transactions on Medical Imaging*, *29*(6), 1310-1320. https://doi.org/10.1109/TMI.2010.2046908

Tustison, N. J., Yassa, M. A., Rizvi, B., Cook, P. A., Holbrook, A. J., Srinivasan, M. T., Duda, J. T., Mangan, J. M., Stone, J. R., Manzanera, E. C., Shah, D., Satterthwaite, T. D., Elliott, M. A., Cieslak, M. C., Sydnor, V. J., Davatzikos, C., Oathes, D. J., Dolui, S., Detre, J. A., ... Avants, B. B. (2024). ANTsX neuroimaging-derived structural phenotypes of UK Biobank. *Scientific Reports*, *14*, 8848. https://doi.org/10.1038/s41598-024-59440-6

Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014). Permutation inference for the general linear model. *NeuroImage*, *92*, 381-397. https://doi.org/10.1016/j.neuroimage.2014.01.060

Zang, Y., Jiang, T., Lu, Y., He, Y., & Tian, L. (2004). Regional homogeneity approach to fMRI data analysis. *NeuroImage*, *22*(1), 394-400. https://doi.org/10.1016/j.neuroimage.2003.12.030

Zou, Q. H., Zhu, C. Z., Yang, Y., Zuo, X. N., Long, X. Y., Cao, Q. J., Wang, Y. F., & Zang, Y. F. (2008). An improved approach to detection of amplitude of low-frequency fluctuation (ALFF) for resting-state fMRI: Fractional ALFF. *Journal of Neuroscience Methods*, *172*(1), 137-141. https://doi.org/10.1016/j.jneumeth.2008.04.012