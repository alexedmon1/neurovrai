# Project Cleanup Plan

## Current State
Project has accumulated legacy scripts and documentation that should be archived.

## Cleanup Strategy

### ✅ KEEP IN ROOT (Production/Current)

**Scripts:**
- `create_config.py` - Production config generator
- `verify_environment.py` - Environment validation
- `run_simple_pipeline.py` - Current production runner
- `run_batch_simple.py` - Current batch processor

**Documentation:**
- `README.md` - Main project documentation
- `QUICKSTART.md` - Quick start guide (primary user doc)
- `CLAUDE.md` - AI assistant guidelines
- `SETUP_GUIDE.md` - Initial setup instructions
- `DEPENDENCIES.md` - Dependency reference
- `PROJECT_STATUS.md` - Current implementation status

### 📦 ARCHIVE

**Scripts → `archive/runners/`:**
- `run_preprocessing.py` - Old production runner (replaced by run_simple_pipeline.py)
- `run_full_pipeline.py` - Complex monitoring version
- `run_continuous_pipeline.py` - Continuous monitoring version
- `run_all_subjects.py` - Old batch runner
- `run_batch_all_subjects.py` - Old batch runner

**Documentation → `docs/archive/`:**
- `CONFIG_SETUP.md` - Detailed config guide (info now in QUICKSTART.md)
- `CONFIG_SUMMARY.md` - Config summary (info now in QUICKSTART.md)
- `SIMPLE_PIPELINE_GUIDE.md` - Pipeline guide (info now in QUICKSTART.md)

### 📁 Current Directory Structure

```
human-mri-preprocess/
├── README.md                    # Main documentation
├── QUICKSTART.md                # Primary user guide
├── SETUP_GUIDE.md               # Initial setup
├── DEPENDENCIES.md              # Dependency reference
├── PROJECT_STATUS.md            # Implementation status
├── CLAUDE.md                    # AI guidelines
├── create_config.py             # Config generator
├── verify_environment.py        # Environment check
├── run_simple_pipeline.py       # Production runner
├── run_batch_simple.py          # Batch processor
├── mri_preprocess/              # Production code
│   ├── workflows/               # Production workflows
│   ├── utils/                   # Helper functions
│   ├── qc/                      # QC modules
│   └── dicom/                   # DICOM converters
├── docs/                        # Documentation
│   ├── README.md                # Docs navigation
│   ├── FUTURE_ENHANCEMENTS.md   # Planned features
│   ├── workflows.md             # Workflow guide
│   ├── implementation/          # Technical details
│   ├── status/                  # Progress tracking
│   └── archive/                 # Old documentation
├── archive/                     # Legacy code
│   ├── runners/                 # Old pipeline runners
│   ├── anat/                    # Legacy anatomical
│   ├── dwi/                     # Legacy diffusion
│   ├── rest/                    # Legacy functional
│   └── tests/                   # Test scripts
└── examples/                    # Usage examples
```

## Benefits

1. **Clear separation**: Production vs legacy code
2. **Easy navigation**: Users find current tools quickly
3. **Preserved history**: Old code archived, not deleted
4. **Reduced confusion**: One current runner, not five options
5. **Clean root**: Only essential files visible
