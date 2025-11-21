#!/usr/bin/env python3
"""
Design Matrix Generation for FSL Group Analysis

Creates FSL-format design matrices (.mat files) from participant data and model formulas.

Supports:
- Continuous predictors (with optional mean-centering/standardization)
- Categorical predictors (dummy coding)
- Interaction terms
- Custom contrast matrices (.con files)
- Automatic subject matching with prepared data

Example formulas:
- "age + sex"                    # Main effects
- "age + sex + exposure"         # Multiple predictors
- "age + sex + age*sex"          # Interaction
- "C(group) + age"               # Explicit categorical coding

Design matrix format (FSL .mat):
    /NumWaves   4
    /NumPoints  50
    /PPheights  1 1 1 1
    /Matrix
    1 45 0 2.3
    1 38 1 5.1
    ...
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class DesignMatrixError(Exception):
    """Raised when design matrix creation fails"""
    pass


def load_participants(
    participants_file: Path,
    subject_list_file: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load participants CSV and optionally filter to match subject list

    Args:
        participants_file: Path to participants.csv
        subject_list_file: Optional path to subject_list.txt from preparation

    Returns:
        DataFrame with participant data

    Raises:
        DesignMatrixError: If file not found or subjects don't match
    """
    if not participants_file.exists():
        raise DesignMatrixError(f"Participants file not found: {participants_file}")

    df = pd.read_csv(participants_file)

    # Check for required column
    if 'subject_id' not in df.columns:
        raise DesignMatrixError(
            "Participants CSV must have 'subject_id' column. "
            f"Found columns: {list(df.columns)}"
        )

    # Filter to subject list if provided
    if subject_list_file is not None:
        if not subject_list_file.exists():
            raise DesignMatrixError(f"Subject list not found: {subject_list_file}")

        with open(subject_list_file, 'r') as f:
            included_subjects = [line.strip() for line in f if line.strip()]

        # Filter DataFrame
        original_count = len(df)
        df = df[df['subject_id'].isin(included_subjects)]

        if len(df) == 0:
            raise DesignMatrixError(
                "No subjects matched between participants CSV and subject list!"
            )

        if len(df) != len(included_subjects):
            missing = set(included_subjects) - set(df['subject_id'])
            raise DesignMatrixError(
                f"Missing {len(missing)} subjects in participants CSV: {missing}"
            )

        # Reorder to match subject list order
        df = df.set_index('subject_id').loc[included_subjects].reset_index()

        logging.info(
            f"Filtered participants: {original_count} → {len(df)} subjects"
        )

    return df


def parse_formula(formula: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Parse model formula into main effects and interactions

    Args:
        formula: Model formula (e.g., "age + sex + age*sex")

    Returns:
        Tuple of (main_effects, interactions)
        - main_effects: List of predictor names
        - interactions: List of (var1, var2) tuples

    Example:
        >>> parse_formula("age + sex + exposure + age*sex")
        (['age', 'sex', 'exposure'], [('age', 'sex')])
    """
    # Remove whitespace
    formula = formula.replace(' ', '')

    # Split by +
    terms = formula.split('+')

    main_effects = []
    interactions = []

    for term in terms:
        if '*' in term:
            # Interaction term
            vars = term.split('*')
            if len(vars) != 2:
                raise DesignMatrixError(
                    f"Only two-way interactions supported: {term}"
                )
            interactions.append((vars[0], vars[1]))
            # Also add main effects if not already present
            for var in vars:
                if var not in main_effects:
                    main_effects.append(var)
        else:
            # Main effect
            if term not in main_effects:
                main_effects.append(term)

    return main_effects, interactions


def create_design_matrix(
    df: pd.DataFrame,
    formula: str,
    demean_continuous: bool = True,
    add_intercept: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Create design matrix from DataFrame and formula

    Args:
        df: DataFrame with participant data
        formula: Model formula (e.g., "age + sex + exposure")
        demean_continuous: Whether to mean-center continuous variables
        add_intercept: Whether to add intercept column

    Returns:
        Tuple of (design_matrix, column_names)
        - design_matrix: NumPy array (n_subjects x n_predictors)
        - column_names: List of predictor names

    Raises:
        DesignMatrixError: If variables not found or invalid types
    """
    main_effects, interactions = parse_formula(formula)

    # Check all variables exist
    for var in main_effects:
        clean_var = var.replace('C(', '').replace(')', '')
        if clean_var not in df.columns:
            raise DesignMatrixError(
                f"Variable '{clean_var}' not found in participants data. "
                f"Available: {list(df.columns)}"
            )

    # Build design matrix column by column
    design_cols = []
    col_names = []

    # Add intercept if requested
    if add_intercept:
        design_cols.append(np.ones(len(df)))
        col_names.append('intercept')

    # Add main effects
    for var in main_effects:
        # Check if explicitly coded as categorical
        is_categorical = var.startswith('C(')
        clean_var = var.replace('C(', '').replace(')', '')

        series = df[clean_var]

        # Determine if categorical or continuous
        if is_categorical or series.dtype == 'object' or series.dtype.name == 'category':
            # Categorical variable - dummy coding
            if series.nunique() > 10:
                logging.warning(
                    f"Variable '{clean_var}' has {series.nunique()} levels. "
                    "Are you sure it's categorical?"
                )

            # Create dummy variables (drop first level to avoid collinearity)
            dummies = pd.get_dummies(series, drop_first=True, dtype=float)

            for col in dummies.columns:
                design_cols.append(dummies[col].values)
                col_names.append(f"{clean_var}_{col}")

        else:
            # Continuous variable
            vals = series.values.astype(float)

            # Check for missing values
            if np.any(np.isnan(vals)):
                raise DesignMatrixError(
                    f"Variable '{clean_var}' has missing values"
                )

            # Optionally mean-center
            if demean_continuous:
                vals = vals - np.mean(vals)

            design_cols.append(vals)
            col_names.append(clean_var)

    # Add interactions
    for var1, var2 in interactions:
        # Find columns for each variable
        var1_cols = [i for i, name in enumerate(col_names) if name.startswith(var1)]
        var2_cols = [i for i, name in enumerate(col_names) if name.startswith(var2)]

        if not var1_cols or not var2_cols:
            raise DesignMatrixError(
                f"Cannot create interaction {var1}*{var2}: variables not in design"
            )

        # Create interaction columns
        for i in var1_cols:
            for j in var2_cols:
                interaction = design_cols[i] * design_cols[j]
                design_cols.append(interaction)
                col_names.append(f"{col_names[i]}*{col_names[j]}")

    # Stack into matrix
    design_matrix = np.column_stack(design_cols)

    return design_matrix, col_names


def write_fsl_design_matrix(
    design_matrix: np.ndarray,
    output_file: Path,
    column_names: Optional[List[str]] = None
):
    """
    Write design matrix in FSL .mat format

    Args:
        design_matrix: Design matrix array (n_subjects x n_predictors)
        output_file: Output path for .mat file
        column_names: Optional predictor names (for logging)
    """
    n_subjects, n_predictors = design_matrix.shape

    with open(output_file, 'w') as f:
        # Header
        f.write(f"/NumWaves\t{n_predictors}\n")
        f.write(f"/NumPoints\t{n_subjects}\n")
        f.write("/PPheights\t" + "\t".join(["1"] * n_predictors) + "\n")
        f.write("\n/Matrix\n")

        # Data
        for row in design_matrix:
            f.write("\t".join([f"{val:.6f}" for val in row]) + "\n")

    logging.info(f"Design matrix saved: {output_file}")
    logging.info(f"  Subjects: {n_subjects}")
    logging.info(f"  Predictors: {n_predictors}")
    if column_names:
        logging.info(f"  Columns: {', '.join(column_names)}")


def create_contrast_matrix(
    contrasts: List[Dict],
    column_names: List[str],
    output_file: Path
):
    """
    Create FSL contrast matrix from contrast specifications

    Args:
        contrasts: List of contrast dictionaries with keys:
            - name: Contrast name
            - vector: List of weights for each predictor
            - type: 'tstat' or 'fstat'
        column_names: Predictor names (for validation)
        output_file: Output path for .con file

    Raises:
        DesignMatrixError: If contrast vectors don't match design matrix
    """
    n_predictors = len(column_names)
    n_contrasts = len(contrasts)

    # Validate contrasts
    for i, contrast in enumerate(contrasts):
        if 'name' not in contrast:
            raise DesignMatrixError(f"Contrast {i} missing 'name' field")
        if 'vector' not in contrast:
            raise DesignMatrixError(f"Contrast '{contrast['name']}' missing 'vector' field")
        if len(contrast['vector']) != n_predictors:
            raise DesignMatrixError(
                f"Contrast '{contrast['name']}' has {len(contrast['vector'])} weights "
                f"but design matrix has {n_predictors} predictors"
            )

    with open(output_file, 'w') as f:
        # Header
        f.write(f"/NumWaves\t{n_predictors}\n")
        f.write(f"/NumContrasts\t{n_contrasts}\n")
        f.write("/PPheights\t" + "\t".join(["1"] * n_contrasts) + "\n")
        f.write("\n/Matrix\n")

        # Data
        for contrast in contrasts:
            weights = contrast['vector']
            f.write("\t".join([f"{w:.6f}" for w in weights]) + "\n")

    logging.info(f"Contrast matrix saved: {output_file}")
    logging.info(f"  Contrasts: {n_contrasts}")
    for contrast in contrasts:
        logging.info(f"    - {contrast['name']}: {contrast['vector']}")


def generate_design_files(
    participants_file: Path,
    formula: str,
    contrasts: List[Dict],
    output_dir: Path,
    subject_list_file: Optional[Path] = None,
    demean_continuous: bool = True,
    add_intercept: bool = True
) -> Dict:
    """
    Main function: Generate FSL design and contrast files

    Args:
        participants_file: Path to participants.csv
        formula: Model formula (e.g., "age + sex + exposure")
        contrasts: List of contrast specifications
        output_dir: Output directory for .mat and .con files
        subject_list_file: Optional subject list from preparation
        demean_continuous: Whether to mean-center continuous variables
        add_intercept: Whether to add intercept

    Returns:
        Dictionary with paths to generated files and metadata

    Example:
        >>> generate_design_files(
        ...     participants_file=Path('participants.csv'),
        ...     formula='age + sex + exposure',
        ...     contrasts=[
        ...         {'name': 'age_positive', 'vector': [0, 1, 0, 0]},
        ...         {'name': 'exposure_negative', 'vector': [0, 0, 0, -1]}
        ...     ],
        ...     output_dir=Path('/study/analysis/tbss_FA/model1/')
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and filter participants
    df = load_participants(participants_file, subject_list_file)

    # Create design matrix
    design_matrix, column_names = create_design_matrix(
        df=df,
        formula=formula,
        demean_continuous=demean_continuous,
        add_intercept=add_intercept
    )

    # Write design matrix
    design_mat_file = output_dir / "design.mat"
    write_fsl_design_matrix(design_matrix, design_mat_file, column_names)

    # Write contrast matrix
    contrast_con_file = output_dir / "design.con"
    create_contrast_matrix(contrasts, column_names, contrast_con_file)

    # Generate design summary
    summary = {
        'n_subjects': len(df),
        'n_predictors': len(column_names),
        'n_contrasts': len(contrasts),
        'formula': formula,
        'column_names': column_names,
        'design_mat_file': str(design_mat_file),
        'contrast_con_file': str(contrast_con_file),
        'demean_continuous': demean_continuous,
        'add_intercept': add_intercept
    }

    # Write summary
    import json
    summary_file = output_dir / "design_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Design summary saved: {summary_file}")

    return summary


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create example participants data
    example_df = pd.DataFrame({
        'subject_id': [f'sub-{i:03d}' for i in range(1, 21)],
        'age': np.random.randint(25, 65, 20),
        'sex': np.random.choice(['M', 'F'], 20),
        'exposure': np.random.uniform(0, 10, 20)
    })

    example_df.to_csv('/tmp/participants_example.csv', index=False)

    # Generate design files
    result = generate_design_files(
        participants_file=Path('/tmp/participants_example.csv'),
        formula='age + sex + exposure',
        contrasts=[
            {'name': 'age_positive', 'vector': [0, 1, 0, 0]},
            {'name': 'sex_MvsF', 'vector': [0, 0, 1, 0]},
            {'name': 'exposure_negative', 'vector': [0, 0, 0, -1]}
        ],
        output_dir=Path('/tmp/design_test/')
    )

    print("\nGenerated files:")
    print(f"  Design matrix: {result['design_mat_file']}")
    print(f"  Contrasts: {result['contrast_con_file']}")
