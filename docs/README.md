# neurovrai Documentation

## Quick Reference

| Document | Description |
|----------|-------------|
| [../README.md](../README.md) | Main project README with usage examples |
| [../CLAUDE.md](../CLAUDE.md) | AI assistant guidelines |
| [../PROJECT_STATUS.md](../PROJECT_STATUS.md) | Current implementation status |

## Core Documentation

### Configuration & Workflows
- [configuration.md](configuration.md) - Configuration file reference
- [workflows.md](workflows.md) - Preprocessing workflow details
- [cli.md](cli.md) - Command-line interface reference
- [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md) - Output directory organization

### Processing Guides
- [DWI_PROCESSING_GUIDE.md](DWI_PROCESSING_GUIDE.md) - Diffusion preprocessing guide
- [DKI_METRICS.md](DKI_METRICS.md) - DKI metrics interpretation
- [GLM_USAGE.md](GLM_USAGE.md) - Group-level GLM analysis
- [DUAL_REGRESSION_USAGE.md](DUAL_REGRESSION_USAGE.md) - MELODIC dual regression
- [MODALITY_AWARE_DESIGN_GENERATION.md](MODALITY_AWARE_DESIGN_GENERATION.md) - Design matrix generation

### Quality Control
- [skull_strip_qc_usage.md](skull_strip_qc_usage.md) - Skull stripping QC guide

### Reference
- [DEPENDENCIES.md](DEPENDENCIES.md) - Package dependencies
- [NEUROVRAI_ARCHITECTURE.md](NEUROVRAI_ARCHITECTURE.md) - System architecture
- [FUTURE_ENHANCEMENTS.md](FUTURE_ENHANCEMENTS.md) - Planned features

## Module Documentation

Each main module has its own README:

- `neurovrai/analysis/README.md` - Analysis module overview
- `neurovrai/connectome/README.md` - Connectome module overview

## Subdirectories

- `amico/` - AMICO-specific documentation
- `analysis/` - Group analysis documentation
- `examples/` - Example configurations
- `implementation/` - Technical implementation details
- `issues/` - Known issues and workarounds
- `archive/` - Archived documentation

## For Users

Start with:
1. `../README.md` - Main project README
2. `configuration.md` - Setup and configuration
3. `workflows.md` - Understanding the pipelines

## For Developers

Technical implementation:
1. `implementation/` - Feature implementation details
2. `NEUROVRAI_ARCHITECTURE.md` - System architecture
3. `../CLAUDE.md` - AI-assisted development guidelines
