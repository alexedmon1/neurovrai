# Regression (behavior-preservation) tests

These tests enforce the **boundary** of an implementation: given a frozen
input, a candidate must reproduce a frozen output within tolerance. They are
the half of the gate that unit tests don't cover — *did my change keep the
results the same?* — and they let a refactor be promoted with confidence.

## Layout

```
tests/regression/
├── _equivalence.py   # helpers: fixture_path, golden_path, assert_array_matches_golden
├── fixtures/         # frozen inputs (small, committed; tracked even if .nii.gz/.mat)
├── golden/           # expected outputs (committed)
└── test_*.py         # @pytest.mark.regression tests
```

## Writing a regression test for a real change

1. Capture a **small** representative input into `fixtures/`.
2. Mint the golden from the *current* (trusted) implementation once:
   ```bash
   NEUROVRAI_UPDATE_GOLDEN=1 uv run pytest -m regression
   ```
   Review and commit the new `golden/*.npy` in its own commit.
3. Write the test so the **candidate** runs against the fixture and is
   compared to the golden:
   ```python
   import pytest
   from tests.regression._equivalence import fixture_path, assert_array_matches_golden

   pytestmark = pytest.mark.regression

   def test_connectivity_matrix_preserved():
       result = build_structural_connectivity(fixture_path("tracts_small.npz"))  # candidate
       assert_array_matches_golden(result, "sc_matrix")
   ```
4. `make check` now fails if any candidate drifts.
