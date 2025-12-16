# Documentation Status Index

**Last Updated:** 2025-12-11

This index helps navigate the various status, roadmap, and planning documents in the neurovrai project.

## 📋 Current Active Documents

### Primary Planning & Roadmap

**[ROADMAP_2025-12.md](ROADMAP_2025-12.md)** ⭐ **PRIMARY REFERENCE**
- **Purpose:** Comprehensive development roadmap with priorities
- **Last Updated:** 2025-12-11
- **Contains:**
  - Project status overview
  - Prioritized task list (Critical/Medium/Low)
  - Implementation plans with time estimates
  - Success metrics and testing strategy
- **Use For:** Planning next development sessions

### Current Session Documentation

**[sessions/SESSION_2025-12-11_FUNCTIONAL_WORKFLOW_FIX.md](sessions/SESSION_2025-12-11_FUNCTIONAL_WORKFLOW_FIX.md)**
- **Purpose:** Detailed session summary for functional preprocessing fix
- **Date:** 2025-12-11
- **Topic:** Moving BET to Phase 1, fixing ApplyXFM4D interface
- **Status:** ✅ Complete

**[FUNCTIONAL_PREPROCESSING_RESOLVED.md](FUNCTIONAL_PREPROCESSING_RESOLVED.md)**
- **Purpose:** Issue resolution document for functional preprocessing
- **Date:** 2025-12-11
- **Resolves:** docs/FUNCTIONAL_PREPROCESSING_ISSUES.md
- **Status:** ✅ Resolved

### Issue Tracking

**[FUNCTIONAL_PREPROCESSING_ISSUES.md](FUNCTIONAL_PREPROCESSING_ISSUES.md)**
- **Purpose:** Root cause analysis of functional preprocessing issues
- **Status:** ✅ RESOLVED (see FUNCTIONAL_PREPROCESSING_RESOLVED.md)
- **Keep For:** Historical reference

**[SESSION_CLEANUP_STRATEGY.md](SESSION_CLEANUP_STRATEGY.md)**
- **Purpose:** Strategy for cleaning up stashed functional connectivity work
- **Date:** 2025-12-08
- **Status:** ⏳ Pending (awaiting functional preprocessing fix completion)

### Future Enhancements

**[FUTURE_ENHANCEMENTS.md](FUTURE_ENHANCEMENTS.md)**
- **Purpose:** Granular tracking of configuration enhancements
- **Focus:** Making parameters configurable via YAML
- **Status:** ✅ Active reference
- **Note:** Complements ROADMAP_2025-12.md with implementation details

### Module-Specific Status

**[analysis/ANALYSIS_PROJECT_STATUS.md](analysis/ANALYSIS_PROJECT_STATUS.md)**
- **Purpose:** Status of analysis module features
- **Scope:** VBM, TBSS, resting-state, connectivity
- **Status:** ✅ Active reference

**[status/DWI_ADVANCED_STATUS.md](status/DWI_ADVANCED_STATUS.md)**
- **Purpose:** DKI and NODDI implementation status
- **Date:** 2025-11-18
- **Status:** ✅ Complete (kept for reference)

**[status/RESTING_STATE_STATUS.md](status/RESTING_STATE_STATUS.md)**
- **Purpose:** ReHo, fALFF, MELODIC status
- **Date:** 2025-11-18
- **Status:** ✅ Complete (kept for reference)

### Operational Status

**[status/OVERNIGHT_RUN_STATUS.md](status/OVERNIGHT_RUN_STATUS.md)**
- **Purpose:** Tracking long-running pipeline tests
- **Status:** ✅ Active reference

**[status/CONTINUOUS_PIPELINE_TEST.md](status/CONTINUOUS_PIPELINE_TEST.md)**
- **Purpose:** Continuous testing setup and results
- **Status:** ✅ Active reference

### Historical Context

**[status/SESSION_HISTORY_2025-11.md](status/SESSION_HISTORY_2025-11.md)**
- **Purpose:** Comprehensive development history for November 2025
- **Contains:** All major features implemented
- **Status:** ✅ Complete (historical reference)

**[status/SESSION_2025-12-05_DUMMY_CODING.md](status/SESSION_2025-12-05_DUMMY_CODING.md)**
- **Purpose:** GLM dummy coding implementation
- **Date:** 2025-12-05
- **Status:** ✅ Complete (historical reference)

---

## 📦 Archived Documents

### Archive Organization

**[archive/sessions-2025-12/](archive/sessions-2025-12/)**
- Old session summaries from December 2025
- Design regeneration documents
- Refactoring progress summaries

**Files:**
- `SESSION_SUMMARY_2025-12-05.md`
- `SESSION_SUMMARY_2025-12-05_PART2.md`
- `SESSION_SUMMARY_2025-12-05_PART3.md`
- `SESSION_STATUS_2025-12-05_END.md`
- `DESIGN_REGENERATION_SUMMARY.md`
- `refactor_progress_summary.md`

**[archive/](archive/)** (general)
- Outdated roadmaps and TODOs
- Superseded documentation
- Historical context

**Files:**
- `AMICO_TODO.md` - Completed AMICO integration
- `DWI_ROADMAP.md` - Old DWI planning (superseded by ROADMAP_2025-12.md)
- Plus various implementation guides (moved to docs/implementation/)

---

## 🔄 Document Lifecycle

### When to Create New Documents

**Session Summaries:**
- Create for significant development sessions (2+ hours of focused work)
- Location: `docs/sessions/SESSION_YYYY-MM-DD_TOPIC.md`
- Template: Include objective, implementation, testing, results

**Issue Resolutions:**
- Create when resolving documented issues
- Location: `docs/[TOPIC]_RESOLVED.md`
- Must reference original issue document

**Status Updates:**
- Module-specific: Update existing files in `docs/status/`
- Project-wide: Update `ROADMAP_2025-12.md`

### When to Archive Documents

**Archive Criteria:**
- Session summaries >1 month old
- Completed implementation plans
- Superseded roadmaps or TODO lists
- One-time analysis or investigation reports

**Archive Process:**
1. Move to `docs/archive/` or `docs/archive/sessions-YYYY-MM/`
2. Update this index
3. Add reference in ROADMAP if relevant

### When to Update Documents

**Update ROADMAP_2025-12.md when:**
- Starting a new major feature
- Completing critical tasks
- Re-prioritizing work
- Monthly review (due: 2026-01-11)

**Update Module Status when:**
- Feature reaches production-ready
- Major bugs fixed
- New capabilities added

---

## 📊 Quick Reference

### I want to...

**Plan next development work:**
→ Read **ROADMAP_2025-12.md** (Critical/High Priority sections)

**Understand recent changes:**
→ Check **sessions/** for latest SESSION_*.md files

**See what's production-ready:**
→ Read **ROADMAP_2025-12.md** (Production-Ready Modules section)

**Find implementation details:**
→ Check **implementation/** directory or FUTURE_ENHANCEMENTS.md

**Track a specific module:**
→ See **status/** or **analysis/** directories

**Review historical work:**
→ Browse **archive/** or **status/SESSION_HISTORY_2025-11.md**

---

## 🎯 Maintenance

### Monthly Review Checklist

- [ ] Update ROADMAP_2025-12.md priorities
- [ ] Archive old session summaries (>1 month)
- [ ] Update module status documents
- [ ] Review and close completed issues
- [ ] Update this index

### Document Hygiene

**Keep:**
- Current roadmap and planning docs
- Active issue tracking
- Recent session summaries (<1 month)
- Module-specific status (production reference)

**Archive:**
- Old session summaries (>1 month)
- Completed implementation plans
- Superseded roadmaps
- One-time investigations

**Delete:**
- Duplicate information
- Obsolete references
- Test/scratch documents

---

**Maintained By:** Development Team
**Review Frequency:** Monthly
**Next Review:** 2026-01-11
