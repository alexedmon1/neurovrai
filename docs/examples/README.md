# MRI Preprocessing Examples

This directory contains practical examples demonstrating how to use the MRI preprocessing pipeline.

## Examples Overview

### 01_basic_preprocessing.py
**Simple single-modality preprocessing**

Shows how to run each modality independently:
- Anatomical (T1w) preprocessing
- DWI preprocessing with TOPUP distortion correction
- Functional preprocessing (multi-echo and single-echo)
- ASL preprocessing with CBF quantification

**Use when:**
- You want to process one modality at a time
- You're learning the pipeline
- You need to re-run a specific modality

**Run it:**
\`\`\`bash
python examples/01_basic_preprocessing.py
\`\`\`

---

### 02_full_pipeline.py
**Complete multi-modality pipeline**

Demonstrates running the full preprocessing pipeline for a single subject:
1. Anatomical preprocessing (required first)
2. DWI preprocessing (independent)
3. Functional preprocessing (depends on anatomical)
4. ASL preprocessing (depends on anatomical)

Includes proper dependency management and error handling.

**Use when:**
- Processing a complete subject with all modalities
- You want to see modality dependencies
- Running quality control across all modalities

**Run it:**
\`\`\`bash
python examples/02_full_pipeline.py
\`\`\`

---

### 03_batch_processing.py
**Batch processing multiple subjects**

Shows how to process an entire study with multiple subjects:
- Automatic subject discovery
- Progress tracking with resumption
- Error handling and logging
- Modality dependencies
- Status reporting

Features:
- **Progress tracking**: Saves status to JSON file
- **Resumption**: Skips already-completed subjects/modalities
- **Error handling**: Continues processing even if subjects fail
- **Summary**: Final report of success/failure rates

**Use when:**
- Processing multiple subjects in a study
- Running overnight/long jobs
- Need to resume after interruption

**Run it:**
\`\`\`bash
python examples/03_batch_processing.py
\`\`\`

Progress is saved to \`logs/batch_progress.json\`. Re-run the script to resume.

---

## Production Command-Line Tools

For production use, the pipeline provides command-line tools in the root directory:

### run_preprocessing.py
Simple CLI for single subjects:
\`\`\`bash
# Run anatomical
python run_preprocessing.py --subject IRC805-0580101 --modality anat

# Run DWI
python run_preprocessing.py --subject IRC805-0580101 --modality dwi

# Run functional
python run_preprocessing.py --subject IRC805-0580101 --modality func

# Run ASL
python run_preprocessing.py --subject IRC805-0580101 --modality asl

# Run all modalities
python run_preprocessing.py --subject IRC805-0580101 --modality all
\`\`\`

### run_full_pipeline.py
Complete orchestrator from DICOM to outputs:
\`\`\`bash
# From DICOM directory
python run_full_pipeline.py \\
    --subject IRC805-0580101 \\
    --dicom-dir /mnt/bytopia/IRC805/dicoms/IRC805-0580101 \\
    --config config.yaml

# From NIfTI directory (skip DICOM conversion)
python run_full_pipeline.py \\
    --subject IRC805-0580101 \\
    --nifti-dir /mnt/bytopia/IRC805/bids/IRC805-0580101 \\
    --config config.yaml
\`\`\`

### run_continuous_pipeline.py
Advanced streaming pipeline:
\`\`\`bash
python run_continuous_pipeline.py \\
    --subject IRC805-0580101 \\
    --dicom-dir /mnt/bytopia/IRC805/dicoms/IRC805-0580101 \\
    --config config.yaml
\`\`\`

Monitors DICOM conversion and starts workflows as files become available.

---

## Configuration

All examples and tools use \`config.yaml\` for settings. See \`docs/configuration.md\` for details.

**Key configuration sections:**
- \`project_dir\`: Study root directory
- \`execution\`: Parallel processing settings
- \`anatomical\`: T1w preprocessing parameters
- \`diffusion\`: DWI/DTI parameters, TOPUP settings
- \`functional\`: fMRI parameters, TEDANA/ICA-AROMA settings
- \`asl\`: ASL/CBF quantification parameters

---

## Output Structure

All preprocessing outputs follow a standardized directory structure:

\`\`\`
{study_root}/
├── derivatives/{subject}/      # Preprocessed data
│   ├── anat/                   # Anatomical outputs
│   ├── dwi/                    # DWI outputs
│   ├── func/                   # Functional outputs
│   └── asl/                    # ASL outputs
├── qc/{subject}/               # Quality control reports
│   ├── anat/
│   ├── dwi/
│   ├── func/
│   └── asl/
└── work/{subject}/             # Temporary processing files
    ├── anat_preprocess/
    ├── dwi_preprocess/
    ├── func_preproc/
    └── asl_preprocess/
\`\`\`

---

## Next Steps

1. **Review documentation**: \`docs/README.md\`
2. **Configure pipeline**: Edit \`config.yaml\`
3. **Run examples**: Start with \`01_basic_preprocessing.py\`
4. **Check QC**: Review HTML reports in \`qc/{subject}/\`
5. **Production**: Use \`run_preprocessing.py\` or batch script

For more information, see:
- \`docs/workflows.md\` - Detailed workflow documentation
- \`docs/cli.md\` - Command-line interface guide
- \`docs/configuration.md\` - Configuration reference
- \`README.md\` - Main project README
