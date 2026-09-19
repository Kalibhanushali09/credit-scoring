from scipy import stats
import numpy as np
import pandas as pd


def calculate_psi(expected, actual, buckets=10):
    if pd.Series(expected).nunique() < 2:
        return 0.0

    expected_cut, bins = pd.qcut(expected, q=buckets, retbins=True, duplicates="drop")
    if len(bins) < 2:
        return 0.0

    actual_cut = pd.cut(actual, bins=bins, include_lowest=True)

    expected_pct = expected_cut.value_counts(normalize=True).sort_index()
    actual_pct = actual_cut.value_counts(normalize=True).sort_index()

    actual_pct = actual_pct.replace(0, 0.0001)
    expected_pct = expected_pct.replace(0, 0.0001)

    psi = ((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum()
    return round(float(psi), 4)


def detect_drift(train_df, new_df, features=None):
    results = []
    if features is None:
        features = [
            col
            for col in train_df.columns
            if train_df[col].dtype in ["int64", "float64"]
            and not str(col).startswith("SK_ID")
        ]

    for col in features:
        if col not in new_df.columns:
            continue
        psi = calculate_psi(train_df[col], new_df[col])
        ks_statistic, ks_pvalue = stats.ks_2samp(train_df[col], new_df[col])
        drifted = psi > 0.2 or ks_pvalue < 0.05
        results.append(
            {
                "feature": col,
                "psi": psi,
                "ks_statistic": ks_statistic,
                "ks_pvalue": ks_pvalue,
                "drifted": drifted,
            }
        )
    return pd.DataFrame(results)