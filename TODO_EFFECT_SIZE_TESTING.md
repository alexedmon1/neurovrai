# TODO: Test Effect Size Implementation

**Date**: 2025-01-12
**Priority**: HIGH
**Added by**: Alex Edmondson

## Testing Required

The Cohen's d effect size calculation implementation (`neurovrai/analysis/stats/effect_size.py`) has been added but needs testing with real randomise outputs before production use.

### Test Cases Needed

1. **Basic Functionality**
   - [ ] Test with actual randomise t-statistic files
   - [ ] Verify Cohen's d conversion formula accuracy
   - [ ] Check output file generation

2. **Different Design Types**
   - [ ] Two-sample design (most common)
   - [ ] One-sample design
   - [ ] Paired design (if applicable)

3. **Edge Cases**
   - [ ] Small sample sizes (n<20)
   - [ ] Unequal group sizes
   - [ ] Missing corrected p-value files

4. **Integration Testing**
   - [ ] Test CLI script with real randomise directory
   - [ ] Verify batch processing of multiple contrasts
   - [ ] Check visualization output

### Test Data Locations

```bash
# Example test command (update with real paths):
python scripts/analysis/calculate_effect_sizes.py \
    --randomise-dir /path/to/your/randomise/results \
    --n1 60 --n2 60 \
    --design-type two_sample \
    --output-dir /tmp/effect_size_test
```

### Validation Steps

1. Compare a few voxels manually:
   - Extract t-value from randomise output
   - Calculate d = t × √(1/n₁ + 1/n₂) manually
   - Compare with generated map values

2. Check summary statistics are reasonable:
   - Mean |d| should typically be < 1.0
   - Distribution should include small/medium/large effects

3. Verify thresholded maps only include p<0.05 voxels

### Notes

- Implementation follows standard formulas from Cohen (1988)
- Assumes randomise was run with TFCE
- May need adjustment for different FSL versions

**STATUS**: Awaiting test data

---

Once tested, move this to archive/ and update docs/implementation/effect_size_maps.md with test results.