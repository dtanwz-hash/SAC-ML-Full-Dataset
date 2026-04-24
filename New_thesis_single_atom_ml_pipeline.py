from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import os
import textwrap
from itertools import product
import joblib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    GridSearchCV,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

XGB_AVAILABLE = True
CATBOOST_AVAILABLE = True
LIGHTGBM_AVAILABLE = True

try:
    from xgboost import XGBRegressor
except Exception:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
except Exception:
    CATBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
except Exception:
    LIGHTGBM_AVAILABLE = False


# =============================================================================
# CONFIG
# =============================================================================

DATA_FILE = "FYP Dataset final.csv"
BASE_OUTPUT_DIR = "output"

SCHEMA = {
    "group_col": "DOI",
    "metadata_cols": ["Source", "Title", "DOI"],
    "predictor_cols": [
        "Metal Atom",
        "Atomic Mass of Metal",
        "Support Material",
        "Support Material Class",
        "Metal Loading",
        "Coord Env",
        "Coord Num",
        "Initial Conc of Catalyst",
        "Pollutant",
        "Pollutant Class",
        "Molar Mass of Pollutant",
        "Initial Conc of Pollutant",
        "Oxidant",
        "Initial Conc of Oxidant",
        "Has non-rad",
        "Has SO4",
        "Has OH",
        "Has O2",
        "Has 1O2",
        "Has h+",
        "Experiment Time",
        "Power of Light",
        "Power of Light Class",
        "AOP technique",
    ],
    "target_cols": [
        "Log(TOF)",
        "Rate Constant",
        "Pollutants degraded per min",
        "Retention Per Cycle",
    ],
    "percentage_cols": [
        "Metal Loading",
        "Degradation Efficiency",
        "Catalyst Reusablility",
    ],
    "numeric_candidate_cols": [
        "Atomic Mass of Metal",
        "Metal Loading",
        "Coord Num",
        "Initial Conc of Catalyst",
        "Normalised Conc of Catalyst",
        "Molar Mass of Pollutant",
        "Initial Conc of Pollutant",
        "Normalised Conc of Pollutant",
        "Initial Conc of Oxidant",
        "Normalised Conc of Oxidant",
        "Has non-rad",
        "Has SO4",
        "Has OH",
        "Has O2",
        "Has 1O2",
        "Has h+",
        "Rate Constant",
        "Normalised Rate Constant ",
        "Degradation Efficiency",
        "Experiment Time",
        "Turnover Rate",
        "Log(TOF)",
        "Pollutants degraded per min",
        "Reusablilty Cycles",
        "Retention Per Cycle",
        "Log_retention",
    ],
}

MIN_ROWS_PER_TARGET = 25
N_SPLITS_CV = 5
USE_HOLDOUT_IF_UNIQUE_DOIS_AT_LEAST = 30
HOLDOUT_TEST_SIZE = 0.2
RANDOM_STATE = 42

# =============================================================================
# FULL-COMBINATION SEARCH CONFIG
# =============================================================================

SEARCHABLE_VARIABLE_ALIASES = {
    "Experiment Time": ["Experiment Time", "Illumination Time"],
    "Pollutant Class": ["Pollutant Class"],
    "Metal Atom": ["Metal Atom"],
    "Support Material Class": ["Support Material Class"],
    "Metal Loading": ["Metal Loading"],
    "Oxidant": ["Oxidant"],
}

# Numeric pool controls
MAX_UNIQUE_NUMERIC_VALUES_PER_VARIABLE = 12
NUMERIC_POOL_STRATEGY_DEFAULT = "quantile"   # options: "unique", "quantile", "nearest"
NUMERIC_NEAREST_VALUES_DEFAULT = 8
MAX_FULL_COMBINATION_ROWS = 100000
RANDOM_SAMPLE_IF_EXCEEDS = True
DEFAULT_TOP_N_SEARCH_RESULTS = 20

# =============================================================================
# UTILITIES
# =============================================================================

def ask_yes_no(question: str, default: str = "Y") -> bool:
    while True:
        reply = input(f"{question} ({default}/{'N' if default == 'Y' else 'Y'}): ").strip().upper()
        if reply == "":
            return default == "Y"
        if reply in {"Y", "N"}:
            return reply == "Y"
        print("Please enter Y or N.")


def ask_int(question: str, default: int) -> int:
    while True:
        reply = input(f"{question} [{default}]: ").strip()
        if reply == "":
            return default
        try:
            val = int(reply)
            if val > 0:
                return val
            print("Please enter a positive whole number.")
        except ValueError:
            print("Please enter a valid whole number.")


def ask_float(question: str, default=None) -> float:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        reply = input(f"{question}{suffix}: ").strip()
        if reply == "" and default is not None:
            return float(default)
        try:
            return float(reply)
        except ValueError:
            print("Please enter a valid number.")


def ask_text(question: str, default=None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    reply = input(f"{question}{suffix}: ").strip()
    if reply == "" and default is not None:
        return default
    return reply

def resolve_searchable_variable_name(df: pd.DataFrame, friendly_name: str) -> str | None:
    """
    Map a user-facing controllable variable name to the actual dataset column name.
    """
    aliases = SEARCHABLE_VARIABLE_ALIASES.get(friendly_name, [friendly_name])
    for col in aliases:
        if col in df.columns:
            return col
    return None


def get_default_value_for_column(df: pd.DataFrame, col: str):
    """
    Safe default for prompting and filling non-controlled predictors.
    """
    if col not in df.columns:
        return None

    if pd.api.types.is_numeric_dtype(df[col]):
        val = df[col].median()
        if pd.isna(val):
            return 0
        return float(val)

    mode = df[col].mode(dropna=True)
    if len(mode) > 0:
        return mode.iloc[0]
    return "Unknown"


def format_changed_variables(baseline_row: dict, candidate_row: dict, compare_cols: list[str]) -> tuple[str, str]:
    """
    Returns:
    - compact changed variable names
    - detailed value changes
    """
    changed_names = []
    changed_details = []

    for col in compare_cols:
        base_val = baseline_row.get(col, np.nan)
        cand_val = candidate_row.get(col, np.nan)

        if pd.isna(base_val) and pd.isna(cand_val):
            continue

        if str(base_val) != str(cand_val):
            changed_names.append(col)
            changed_details.append(f"{col}: {base_val} -> {cand_val}")

    return ", ".join(changed_names), " | ".join(changed_details)

def build_candidate_pool_for_variable(
    df: pd.DataFrame,
    col: str,
    baseline_value=None,
    max_unique_numeric_values: int = MAX_UNIQUE_NUMERIC_VALUES_PER_VARIABLE,
    numeric_strategy: str = NUMERIC_POOL_STRATEGY_DEFAULT,
    nearest_k: int = NUMERIC_NEAREST_VALUES_DEFAULT,
):
    """
    For categorical variables:
        - return sorted unique observed non-null values

    For numeric variables:
        - if few unique values: return all sorted unique observed values
        - otherwise use one of:
            * 'unique'   -> first capped sorted unique values
            * 'quantile' -> quantile-based subsample
            * 'nearest'  -> nearest observed values around baseline
    """
    if col not in df.columns:
        return []

    series = df[col].dropna()
    if len(series) == 0:
        return []

    if not pd.api.types.is_numeric_dtype(df[col]):
        vals = sorted(series.astype(str).dropna().unique().tolist())
        if baseline_value is not None and str(baseline_value) not in vals:
            vals.append(str(baseline_value))
            vals = sorted(vals)
        return vals

    # Numeric case
    vals = pd.to_numeric(series, errors="coerce").dropna().unique().tolist()
    vals = sorted(vals)

    if len(vals) <= max_unique_numeric_values:
        if baseline_value is not None and baseline_value not in vals:
            vals.append(float(baseline_value))
            vals = sorted(set(vals))
        return vals

    if numeric_strategy == "unique":
        trimmed = vals[:max_unique_numeric_values]

    elif numeric_strategy == "nearest":
        if baseline_value is None or pd.isna(baseline_value):
            baseline_value = float(np.median(vals))
        distances = sorted([(abs(v - float(baseline_value)), v) for v in vals], key=lambda x: x[0])
        trimmed = sorted([v for _, v in distances[:nearest_k]])

    else:  # default: quantile
        quantile_positions = np.linspace(0, 1, max_unique_numeric_values)
        trimmed = sorted(pd.Series(vals).quantile(quantile_positions).round(10).unique().tolist())

        # snap quantiles back to nearest observed values
        snapped = []
        for q in trimmed:
            nearest = min(vals, key=lambda x: abs(x - q))
            snapped.append(nearest)
        trimmed = sorted(set(snapped))

    if baseline_value is not None and not pd.isna(baseline_value):
        trimmed = sorted(set(trimmed + [float(baseline_value)]))

    return trimmed

def safe_name(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace("+", "plus")
        .replace("-", "_")
    )


@dataclass
class TrainingArtifact:
    target: str
    best_model_name: str
    best_params: dict
    cv_mae_mean: float
    cv_mae_std: float
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_r2_mean: float
    cv_r2_std: float
    holdout_mae: float | None
    holdout_rmse: float | None
    holdout_r2: float | None
    n_rows: int
    n_unique_dois: int
    final_pipeline: Pipeline


# =============================================================================
# DATA LOADING AND CLEANING
# =============================================================================

def load_and_clean_data(data_file: str) -> pd.DataFrame:
    df = pd.read_csv(data_file)
    df.columns = [col.strip() for col in df.columns]

    # Standardize missing values
    missing_tokens = ["NIL", "nil", "Nil", "NA", "na", "Na", "", " "]
    df = df.replace(missing_tokens, np.nan)

    # Remove % where needed
    for col in SCHEMA["percentage_cols"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("%", "", regex=False),
                errors="coerce"
            )

    # Coerce designated columns to numeric
    for col in SCHEMA["numeric_candidate_cols"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strip text fields
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["nan", "None", ""]), col] = np.nan

    binary_indicator_cols = [
    "Has non-rad",
    "Has SO4",
    "Has OH",
    "Has O2",
    "Has 1O2",
    "Has h+",
]

    for col in binary_indicator_cols:
        if col in df.columns:
            df[col] = df[col].astype("Int64").astype(str)
            df[col] = df[col].replace({"<NA>": np.nan})
    
    return df


def validate_schema(df: pd.DataFrame) -> dict:
    report = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "missing_required": [],
        "active_predictors": [c for c in SCHEMA["predictor_cols"] if c in df.columns],
        "active_targets": [c for c in SCHEMA["target_cols"] if c in df.columns],
        "group_col_present": SCHEMA["group_col"] in df.columns,
    }

    if not report["group_col_present"]:
        report["missing_required"].append(SCHEMA["group_col"])

    if len(report["active_predictors"]) == 0:
        report["missing_required"].append("No predictor columns found")

    if len(report["active_targets"]) == 0:
        report["missing_required"].append("No target columns found")

    return report


def create_output_dir(base_dir: str = BASE_OUTPUT_DIR) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base_dir) / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# =============================================================================
# EDA
# =============================================================================

def run_eda(df: pd.DataFrame, output_dir: Path, predictor_cols: list[str], target_cols: list[str]) -> None:
    
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 300

    
    
    def cramers_v(x: pd.Series, y: pd.Series) -> float:
        tab = pd.crosstab(x, y)
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            return np.nan
        chi2 = chi2_contingency(tab)[0]
        n = tab.to_numpy().sum()
        if n <= 1:
            return np.nan
        r, k = tab.shape
        phi2 = chi2 / n
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        denom = min((kcorr - 1), (rcorr - 1))
        if denom <= 0:
            return np.nan
        return float(np.sqrt(phi2corr / denom))

    # Basic summary
    summary_rows = [
        {"Metric": "Rows", "Value": len(df)},
        {"Metric": "Columns", "Value": df.shape[1]},
        {"Metric": "Unique DOIs", "Value": df[SCHEMA["group_col"]].nunique(dropna=True) if SCHEMA["group_col"] in df.columns else np.nan},
        {"Metric": "Predictors used", "Value": len(predictor_cols)},
        {"Metric": "Targets found", "Value": len(target_cols)},
    ]
    pd.DataFrame(summary_rows).to_csv(output_dir / "eda_summary.csv", index=False)

    # Missing summary
    cols = [c for c in predictor_cols + target_cols + [SCHEMA["group_col"]] if c in df.columns]
    missing_df = pd.DataFrame({
        "Column": cols,
        "MissingCount": [df[c].isna().sum() for c in cols],
    })
    missing_df["MissingPercent"] = missing_df["MissingCount"] / len(df) * 100
    missing_df = missing_df.sort_values("MissingPercent", ascending=False)
    missing_df.to_csv(output_dir / "eda_missing_summary.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=missing_df.head(20), x="MissingPercent", y="Column", color="#4C78A8")
    plt.title("Missing data summary")
    plt.xlabel("Missing (%)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_missing_data.png", bbox_inches="tight")
    plt.close()

    # Rows per DOI
    if SCHEMA["group_col"] in df.columns:
        doi_counts = df[SCHEMA["group_col"]].value_counts(dropna=True).reset_index()
        doi_counts.columns = ["DOI", "RowsPerDOI"]
        doi_counts.to_csv(output_dir / "eda_doi_row_counts.csv", index=False)

        plt.figure(figsize=(8, 5))
        sns.histplot(doi_counts["RowsPerDOI"], bins=min(25, max(5, doi_counts["RowsPerDOI"].nunique())), color="#F58518")
        plt.title("Distribution of rows per DOI")
        plt.xlabel("Rows per DOI")
        plt.ylabel("Count of DOIs")
        plt.tight_layout()
        plt.savefig(output_dir / "eda_doi_distribution.png", bbox_inches="tight")
        plt.close()

    # Target distributions
    if target_cols:
        target_long = df[target_cols].melt(var_name="Target", value_name="Value").dropna()
        target_long.to_csv(output_dir / "eda_target_long.csv", index=False)

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=target_long, x="Target", y="Value", color="#72B7B2")
        plt.xticks(rotation=20, ha="right")
        plt.title("Distribution of target variables")
        plt.tight_layout()
        plt.savefig(output_dir / "eda_target_boxplots.png", bbox_inches="tight")
        plt.close()

    # Numeric correlation
    numeric_cols = [c for c in predictor_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        corr.to_csv(output_dir / "eda_numeric_correlation.csv")
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
        plt.title("Numeric predictor correlation heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "eda_numeric_correlation.png", bbox_inches="tight")
        plt.close()

        # Categorical association (Cramer's V) using all categorical predictor columns
    categorical_cols = [
        c for c in predictor_cols
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_cols = [c for c in categorical_cols if c not in ["DOI", "Title", "Source"]]

    if len(categorical_cols) >= 2:
        cv_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols, dtype=float)
        for c1 in categorical_cols:
            for c2 in categorical_cols:
                if c1 == c2:
                    cv_matrix.loc[c1, c2] = 1.0
                else:
                    valid = df[[c1, c2]].dropna()
                    cv_matrix.loc[c1, c2] = cramers_v(valid[c1], valid[c2]) if len(valid) else np.nan

        cv_matrix.to_csv(output_dir / "eda_categorical_cramers_v.csv")

        plt.figure(figsize=(max(8, len(categorical_cols) * 0.7), max(6, len(categorical_cols) * 0.7)))
        sns.heatmap(
            cv_matrix.astype(float),
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            square=True
        )
        plt.title("Categorical association heatmap (Cramer's V)")
        plt.tight_layout()
        plt.savefig(output_dir / "eda_categorical_cramers_v.png", bbox_inches="tight")
        plt.close()

    # Category counts
    for col in categorical_cols:
        if col in df.columns:
            counts = df[col].value_counts(dropna=False).head(20).reset_index()
            counts.columns = [col, "Count"]
            counts.to_csv(output_dir / f"eda_counts_{safe_name(col)}.csv", index=False)

            plt.figure(figsize=(8, 5))
            sns.barplot(data=counts, x="Count", y=col, color="#54A24B")
            plt.title(f"{col} counts")
            plt.tight_layout()
            plt.savefig(output_dir / f"eda_counts_{safe_name(col)}.png", bbox_inches="tight")
            plt.close()


# =============================================================================
# PREPROCESSING
# =============================================================================

def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor, numeric_features, categorical_features


# =============================================================================
# MODELS
# =============================================================================

def get_models_and_grids() -> dict:
    models = {
        "DecisionTree": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {
                "model__max_depth": [4, 6, 8, None],
                "model__min_samples_leaf": [1, 2, 4, 8],
            },
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 8, 12],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "ExtraTrees": (
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 8, 12],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = (
            XGBRegressor(
                random_state=RANDOM_STATE,
                objective="reg:squarederror",
                n_jobs=-1,
                verbosity=0,
            ),
            {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [3, 5, 7],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
        )

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = (
            LGBMRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__num_leaves": [15, 31, 63],
            },
        )

    if CATBOOST_AVAILABLE:
        models["CatBoost"] = (
            CatBoostRegressor(random_state=RANDOM_STATE, verbose=0),
            {
                "model__iterations": [200, 400],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__depth": [4, 6, 8],
            },
        )

    return models


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def compute_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def make_group_splits(groups: pd.Series, n_splits: int = N_SPLITS_CV):
    unique_groups = pd.Series(groups).dropna().unique()
    effective_splits = min(n_splits, len(unique_groups))
    if effective_splits < 2:
        raise ValueError("Not enough unique DOI groups for grouped cross-validation.")
    return GroupKFold(n_splits=effective_splits)


def train_one_target(
    df: pd.DataFrame,
    predictor_cols: list[str],
    target: str,
    output_dir: Path
) -> tuple[pd.DataFrame, TrainingArtifact | None, pd.DataFrame | None]:
    temp = df[predictor_cols + [target, SCHEMA["group_col"]]].copy()
    temp = temp.dropna(subset=[target, SCHEMA["group_col"]])

    if len(temp) < MIN_ROWS_PER_TARGET:
        return pd.DataFrame(), None, None

    # Need enough DOI groups
    if temp[SCHEMA["group_col"]].nunique() < 5:
        return pd.DataFrame(), None, None

    X = temp[predictor_cols].copy()
    y = temp[target].copy()
    groups = temp[SCHEMA["group_col"]].copy()

    # Optional DOI-disjoint holdout
    use_holdout = groups.nunique() >= USE_HOLDOUT_IF_UNIQUE_DOIS_AT_LEAST
    holdout_metrics = None

    if use_holdout:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=HOLDOUT_TEST_SIZE,
            random_state=RANDOM_STATE
        )
        train_idx, holdout_idx = next(splitter.split(X, y, groups))
        X_train, X_holdout = X.iloc[train_idx], X.iloc[holdout_idx]
        y_train, y_holdout = y.iloc[train_idx], y.iloc[holdout_idx]
        groups_train = groups.iloc[train_idx]
    else:
        X_train, y_train, groups_train = X, y, groups
        X_holdout, y_holdout = None, None

    preprocessor, _, _ = build_preprocessor(X_train)
    cv = make_group_splits(groups_train, n_splits=N_SPLITS_CV)
    models = get_models_and_grids()

    benchmark_rows = []
    best_artifact = None
    best_holdout_r2 = -np.inf
    best_cv_r2 = -np.inf
    best_pipeline = None
    best_model_name = None
    best_params = None

    for model_name, (model, grid) in models.items():
        base_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train, groups=groups_train)

        # Fold-level metrics from cv_results are not directly all metrics,
        # so compute grouped CV manually using the best params.
        best_estimator = search.best_estimator_
        fold_metrics = []
        for fold_train_idx, fold_valid_idx in cv.split(X_train, y_train, groups_train):
            est = clone(best_estimator)
            est.fit(X_train.iloc[fold_train_idx], y_train.iloc[fold_train_idx])
            preds = est.predict(X_train.iloc[fold_valid_idx])
            fold_metrics.append(compute_metrics(y_train.iloc[fold_valid_idx], preds))

        cv_mae = [m["MAE"] for m in fold_metrics]
        cv_rmse = [m["RMSE"] for m in fold_metrics]
        cv_r2 = [m["R2"] for m in fold_metrics]

        hold_mae = None
        hold_rmse = None
        hold_r2 = None
        if use_holdout:
            hold_preds = best_estimator.predict(X_holdout)
            hold = compute_metrics(y_holdout, hold_preds)
            hold_mae = hold["MAE"]
            hold_rmse = hold["RMSE"]
            hold_r2 = hold["R2"]

        benchmark_rows.append({
            "Target": target,
            "Model": model_name,
            "RowsUsed": len(temp),
            "UniqueDOIs": temp[SCHEMA["group_col"]].nunique(),
            "HoldoutUsed": use_holdout,
            "CV_MAE_Mean": np.mean(cv_mae),
            "CV_MAE_SD": np.std(cv_mae, ddof=1) if len(cv_mae) > 1 else 0.0,
            "CV_RMSE_Mean": np.mean(cv_rmse),
            "CV_RMSE_SD": np.std(cv_rmse, ddof=1) if len(cv_rmse) > 1 else 0.0,
            "CV_R2_Mean": np.mean(cv_r2),
            "CV_R2_SD": np.std(cv_r2, ddof=1) if len(cv_r2) > 1 else 0.0,
            "Holdout_MAE": hold_mae,
            "Holdout_RMSE": hold_rmse,
            "Holdout_R2": hold_r2,
            "BestParams": json.dumps(search.best_params_),
        })

        better = False
        if use_holdout:
            if hold_r2 is not None and hold_r2 > best_holdout_r2:
                better = True
        else:
            if np.mean(cv_r2) > best_cv_r2:
                better = True

        if better:
            best_holdout_r2 = hold_r2 if hold_r2 is not None else best_holdout_r2
            best_cv_r2 = np.mean(cv_r2)
            best_model_name = model_name
            best_params = search.best_params_
            best_pipeline = best_estimator

            best_artifact = TrainingArtifact(
                target=target,
                best_model_name=model_name,
                best_params=search.best_params_,
                cv_mae_mean=float(np.mean(cv_mae)),
                cv_mae_std=float(np.std(cv_mae, ddof=1) if len(cv_mae) > 1 else 0.0),
                cv_rmse_mean=float(np.mean(cv_rmse)),
                cv_rmse_std=float(np.std(cv_rmse, ddof=1) if len(cv_rmse) > 1 else 0.0),
                cv_r2_mean=float(np.mean(cv_r2)),
                cv_r2_std=float(np.std(cv_r2, ddof=1) if len(cv_r2) > 1 else 0.0),
                holdout_mae=None if hold_mae is None else float(hold_mae),
                holdout_rmse=None if hold_rmse is None else float(hold_rmse),
                holdout_r2=None if hold_r2 is None else float(hold_r2),
                n_rows=len(temp),
                n_unique_dois=int(temp[SCHEMA["group_col"]].nunique()),
                final_pipeline=best_estimator,
            )

    benchmark_df = pd.DataFrame(benchmark_rows).sort_values(
        ["Target", "Holdout_R2" if use_holdout else "CV_R2_Mean"],
        ascending=[True, False]
    )

    # Refit final best model on all usable rows for downstream prediction
    if best_artifact is not None and best_model_name is not None:
        final_preprocessor, _, _ = build_preprocessor(X)
        model_obj, _ = get_models_and_grids()[best_model_name]
        final_pipeline = Pipeline(steps=[
            ("preprocessor", final_preprocessor),
            ("model", model_obj),
        ])
        final_pipeline.set_params(**best_params)
        final_pipeline.fit(X, y)
        best_artifact.final_pipeline = final_pipeline

        # Permutation importance
        if use_holdout:
            imp_X, imp_y = X_holdout, y_holdout
            dataset_used = "Holdout"
        else:
            imp_X, imp_y = X, y
            dataset_used = "FullData_NoExternalHoldout"

        try:
            perm = permutation_importance(
                final_pipeline,
                imp_X,
                imp_y,
                n_repeats=20,
                random_state=RANDOM_STATE,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )
            feature_names = final_pipeline.named_steps["preprocessor"].get_feature_names_out()
            importance_df = pd.DataFrame({
                "Target": target,
                "BestModel": best_model_name,
                "DatasetUsed": dataset_used,
                "Feature": feature_names,
                "ImportanceMean": perm.importances_mean,
                "ImportanceSD": perm.importances_std,
            }).sort_values("ImportanceMean", ascending=False)
        except Exception:
            importance_df = None
    else:
        importance_df = None

    return benchmark_df, best_artifact, importance_df


def build_target_diagnostics(
    df: pd.DataFrame,
    predictor_cols: list[str],
    artifact: TrainingArtifact,
    output_dir: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:

    sns.set_theme(style="whitegrid")
    target = artifact.target
    temp = df[predictor_cols + [target, SCHEMA["group_col"]]].copy().dropna(subset=[target, SCHEMA["group_col"]])
    if temp.empty:
        return None, None

    X = temp[predictor_cols].copy()
    y = temp[target].copy()
    groups = temp[SCHEMA["group_col"]].copy()
    cv = make_group_splits(groups, n_splits=N_SPLITS_CV)

    # Out-of-fold predictions for diagnostic plots
    oof_pred = pd.Series(index=temp.index, dtype=float)
    for tr_idx, va_idx in cv.split(X, y, groups):
        est = clone(artifact.final_pipeline)
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_pred.iloc[va_idx] = est.predict(X.iloc[va_idx])

    diag_df = pd.DataFrame({
        "Observed": y.values,
        "Predicted": oof_pred.values,
    })
    diag_df["Residual"] = diag_df["Observed"] - diag_df["Predicted"]
    diag_df.to_csv(output_dir / f"diagnostics_{safe_name(target)}_oof_predictions.csv", index=False)

    # Predicted vs actual
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=diag_df, x="Observed", y="Predicted")
    mn = np.nanmin([diag_df["Observed"].min(), diag_df["Predicted"].min()])
    mx = np.nanmax([diag_df["Observed"].max(), diag_df["Predicted"].max()])
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.title(f"Predicted vs Actual: {target}")
    plt.tight_layout()
    plt.savefig(output_dir / f"diagnostics_{safe_name(target)}_predicted_vs_actual.png", bbox_inches="tight")
    plt.close()

    # Residual plot
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=diag_df, x="Predicted", y="Residual")
    plt.axhline(0, linestyle="--")
    plt.title(f"Residual Plot: {target}")
    plt.tight_layout()
    plt.savefig(output_dir / f"diagnostics_{safe_name(target)}_residual_plot.png", bbox_inches="tight")
    plt.close()

    # Learning curve
    train_sizes_abs, train_scores, valid_scores = learning_curve(
        clone(artifact.final_pipeline),
        X, y,
        groups=groups,
        cv=cv,
        train_sizes=np.linspace(0.2, 1.0, 5),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        shuffle=False,
    )
    learning_df = pd.DataFrame({
        "TrainSize": train_sizes_abs,
        "TrainRMSE_Mean": -train_scores.mean(axis=1),
        "TrainRMSE_SD": train_scores.std(axis=1),
        "ValidationRMSE_Mean": -valid_scores.mean(axis=1),
        "ValidationRMSE_SD": valid_scores.std(axis=1),
    })
    learning_df.to_csv(output_dir / f"diagnostics_{safe_name(target)}_learning_curve.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(learning_df["TrainSize"], learning_df["TrainRMSE_Mean"], marker="o", label="Train RMSE")
    plt.plot(learning_df["TrainSize"], learning_df["ValidationRMSE_Mean"], marker="o", label="Validation RMSE")
    plt.fill_between(learning_df["TrainSize"],
                     learning_df["TrainRMSE_Mean"] - learning_df["TrainRMSE_SD"],
                     learning_df["TrainRMSE_Mean"] + learning_df["TrainRMSE_SD"],
                     alpha=0.15)
    plt.fill_between(learning_df["TrainSize"],
                     learning_df["ValidationRMSE_Mean"] - learning_df["ValidationRMSE_SD"],
                     learning_df["ValidationRMSE_Mean"] + learning_df["ValidationRMSE_SD"],
                     alpha=0.15)
    plt.title(f"Learning Curve: {target}")
    plt.xlabel("Training rows used")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"diagnostics_{safe_name(target)}_learning_curve.png", bbox_inches="tight")
    plt.close()

    # Intrinsic feature importance when available
    raw_importance_df = None
    try:
        model = artifact.final_pipeline.named_steps["model"]
        raw_importances = getattr(model, "feature_importances_", None)
        if raw_importances is not None:
            feature_names = artifact.final_pipeline.named_steps["preprocessor"].get_feature_names_out()
            raw_importance_df = pd.DataFrame({
                "Target": target,
                "BestModel": artifact.best_model_name,
                "Feature": feature_names,
                "Importance": raw_importances,
            }).sort_values("Importance", ascending=False)
            raw_importance_df.to_csv(output_dir / f"feature_importance_raw_{safe_name(target)}.csv", index=False)
            raw_importance_df.head(20).to_csv(output_dir / f"feature_importance_raw_top20_{safe_name(target)}.csv", index=False)

            plt.figure(figsize=(8, 6))
            sns.barplot(data=raw_importance_df.head(20), x="Importance", y="Feature")
            plt.title(f"Top 20 Model Feature Importances: {target}")
            plt.tight_layout()
            plt.savefig(output_dir / f"feature_importance_raw_top20_{safe_name(target)}.png", bbox_inches="tight")
            plt.close()
    except Exception:
        raw_importance_df = None

    return diag_df, raw_importance_df

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def benchmark_all_targets(df: pd.DataFrame, predictor_cols: list[str], target_cols: list[str], output_dir: Path):
    all_benchmark = []
    best_rows = []
    importances = []
    artifacts = {}

    for target in target_cols:
        bench_df, artifact, imp_df = train_one_target(df, predictor_cols, target, output_dir)
        if not bench_df.empty:
            all_benchmark.append(bench_df)
        if artifact is not None:
            artifacts[target] = artifact
            best_rows.append({
                "Target": artifact.target,
                "BestModel": artifact.best_model_name,
                "RowsUsed": artifact.n_rows,
                "UniqueDOIs": artifact.n_unique_dois,
                "CV_MAE_Mean": artifact.cv_mae_mean,
                "CV_MAE_SD": artifact.cv_mae_std,
                "CV_RMSE_Mean": artifact.cv_rmse_mean,
                "CV_RMSE_SD": artifact.cv_rmse_std,
                "CV_R2_Mean": artifact.cv_r2_mean,
                "CV_R2_SD": artifact.cv_r2_std,
                "Holdout_MAE": artifact.holdout_mae,
                "Holdout_RMSE": artifact.holdout_rmse,
                "Holdout_R2": artifact.holdout_r2,
                "BestParams": json.dumps(artifact.best_params),
            })
        if imp_df is not None and not imp_df.empty:
            importances.append(imp_df)

    benchmark_df = pd.concat(all_benchmark, ignore_index=True) if all_benchmark else pd.DataFrame()
    best_df = pd.DataFrame(best_rows)
    importance_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()

    if not benchmark_df.empty:
        benchmark_df.to_csv(output_dir / "model_benchmark_results.csv", index=False)
    if not best_df.empty:
        best_df.to_csv(output_dir / "best_model_per_target.csv", index=False)
    if not importance_df.empty:
        importance_df.to_csv(output_dir / "permutation_importance_all.csv", index=False)
        top20_perm = importance_df.groupby(["Target", "BestModel"], as_index=False).head(20)
        top20_perm.to_csv(output_dir / "permutation_importance_top20.csv", index=False)

        try:
            for target, group in top20_perm.groupby("Target"):
                plt.figure(figsize=(8, 6))
                sns.barplot(data=group.sort_values("ImportanceMean", ascending=False), x="ImportanceMean", y="Feature")
                plt.title(f"Top 20 Permutation Importances: {target}")
                plt.tight_layout()
                plt.savefig(output_dir / f"permutation_importance_top20_{safe_name(target)}.png", bbox_inches="tight")
                plt.close()
        except Exception:
            pass

    raw_importance_frames = []
    diagnostics_dir = ensure_dir(output_dir / "diagnostics")
    for target, artifact in artifacts.items():
        _, raw_imp_df = build_target_diagnostics(df, predictor_cols, artifact, diagnostics_dir)
        if raw_imp_df is not None and not raw_imp_df.empty:
            raw_importance_frames.append(raw_imp_df)

    if raw_importance_frames:
        raw_all = pd.concat(raw_importance_frames, ignore_index=True)
        raw_all.to_csv(output_dir / "feature_importance_raw_all.csv", index=False)
        raw_all.groupby(["Target", "BestModel"], as_index=False).head(20).to_csv(
            output_dir / "feature_importance_raw_top20.csv", index=False
        )

    return benchmark_df, best_df, importance_df, artifacts


# =============================================================================
# CANDIDATE SCREENING AND PREDICTION
# =============================================================================

def make_candidate_table(
    df: pd.DataFrame,
    predictor_cols: list[str],
    pollutant_class_filter: str | None = None,
    top_n: int = 20
) -> pd.DataFrame:
    cand = df[predictor_cols].copy().drop_duplicates().reset_index(drop=True)

    if pollutant_class_filter is not None and "Pollutant Class" in cand.columns:
        cand = cand[cand["Pollutant Class"] == pollutant_class_filter].copy()

    if cand.empty:
        raise ValueError("Candidate table is empty after filtering.")

    # Keep reasonably sized pool
    max_pool = max(top_n * 50, 500)
    if len(cand) > max_pool:
        cand = cand.sample(max_pool, random_state=RANDOM_STATE).reset_index(drop=True)

    return cand


def screen_candidates(
    df: pd.DataFrame,
    artifacts: dict[str, TrainingArtifact],
    predictor_cols: list[str],
    target: str,
    output_dir: Path,
    top_n: int = 20,
    pollutant_class_filter: str | None = None,
    file_name: str = "candidate_screening.csv"
) -> pd.DataFrame:
    if target not in artifacts:
        raise ValueError(f"No trained model available for target '{target}'.")

    artifact = artifacts[target]
    candidates = make_candidate_table(
        df=df,
        predictor_cols=predictor_cols,
        pollutant_class_filter=pollutant_class_filter,
        top_n=top_n,
    )
    preds = artifact.final_pipeline.predict(candidates)
    candidates = candidates.copy()
    candidates.insert(0, "Target", target)
    candidates.insert(1, "BestModel", artifact.best_model_name)
    candidates["PredictedTarget"] = preds
    candidates = candidates.sort_values("PredictedTarget", ascending=False).head(top_n)
    candidates.to_csv(output_dir / file_name, index=False)
    return candidates


def prompt_for_single_prediction(
    df: pd.DataFrame,
    artifacts: dict[str, TrainingArtifact],
    predictor_cols: list[str],
    target: str,
    output_dir: Path
) -> pd.DataFrame:
    if target not in artifacts:
        raise ValueError(f"No trained model available for target '{target}'.")

    model = artifacts[target].final_pipeline
    input_row = {}

    for col in predictor_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            default = df[col].median()
            if pd.isna(default):
                default = 0
            input_row[col] = ask_float(f"Enter {col}", default=round(float(default), 4))
        else:
            mode = df[col].mode(dropna=True)
            default = mode.iloc[0] if len(mode) > 0 else "Unknown"
            input_row[col] = ask_text(f"Enter {col}", default=str(default))

    input_df = pd.DataFrame([input_row])
    pred = model.predict(input_df)[0]

    result = input_df.copy()
    result.insert(0, "Target", target)
    result.insert(1, "BestModel", artifacts[target].best_model_name)
    result["PredictedTarget"] = pred
    result.to_csv(output_dir / "single_custom_prediction.csv", index=False)
    return result


# =============================================================================
# REPORTING HELPERS
# =============================================================================

def save_schema_and_run_info(df: pd.DataFrame, predictor_cols: list[str], target_cols: list[str], output_dir: Path):
    run_info = {
        "data_file": DATA_FILE,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_unique_dois": int(df[SCHEMA["group_col"]].nunique(dropna=True)) if SCHEMA["group_col"] in df.columns else None,
        "group_col": SCHEMA["group_col"],
        "predictor_cols_used": predictor_cols,
        "target_cols_found": target_cols,
        "random_state": RANDOM_STATE,
        "n_splits_cv": N_SPLITS_CV,
        "holdout_if_unique_dois_at_least": USE_HOLDOUT_IF_UNIQUE_DOIS_AT_LEAST,
        "holdout_test_size": HOLDOUT_TEST_SIZE,
        "xgboost_available": XGB_AVAILABLE,
        "catboost_available": CATBOOST_AVAILABLE,
        "lightgbm_available": LIGHTGBM_AVAILABLE,
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    pd.DataFrame({"ColumnsInDataset": df.columns.tolist()}).to_csv(output_dir / "dataset_columns.csv", index=False)


def save_target_coverage(df: pd.DataFrame, target_cols: list[str], output_dir: Path):
    rows = []
    for target in target_cols:
        temp = df[[SCHEMA["group_col"], target]].copy().dropna(subset=[target])
        rows.append({
            "Target": target,
            "RowsWithTarget": len(temp),
            "UniqueDOIsWithTarget": temp[SCHEMA["group_col"]].nunique(dropna=True),
            "MissingFraction": df[target].isna().mean(),
        })
    pd.DataFrame(rows).to_csv(output_dir / "target_coverage_summary.csv", index=False)

def prompt_for_baseline_experiment(
    df: pd.DataFrame,
    predictor_cols: list[str],
    controllable_cols: list[str],
):
    baseline_values = {}

    print("\nEnter baseline experiment values.")
    print("Leave blank to use a dataset-based default (median for numeric, mode for categorical).")

    # First: controllable variables
    for col in controllable_cols:
        if col not in predictor_cols or col not in df.columns:
            continue

        default_val = get_default_value_for_column(df, col)

        if pd.api.types.is_numeric_dtype(df[col]):
            baseline_values[col] = ask_float(f"Baseline {col}", default=default_val)
        else:
            baseline_values[col] = ask_text(f"Baseline {col}", default=str(default_val))

    # Then: all remaining predictors needed by the model
    remaining_predictors = [c for c in predictor_cols if c not in baseline_values]

    print("\nNow fill remaining predictor fields required by the model.")
    print("You can press Enter to accept the suggested default.")

    for col in remaining_predictors:
        default_val = get_default_value_for_column(df, col)

        if pd.api.types.is_numeric_dtype(df[col]):
            baseline_values[col] = ask_float(f"Baseline {col}", default=default_val)
        else:
            baseline_values[col] = ask_text(f"Baseline {col}", default=str(default_val))

    original_reported_target = ask_text(
        "Optional: enter original reported target/reference value (or leave blank)",
        default=""
    ).strip()

    return baseline_values, original_reported_target

def prompt_for_fixed_and_mutable_variables(controllable_cols: list[str]):
    """
    For each controllable variable, ask whether it is fixed or mutable.
    """
    fixed_vars = []
    mutable_vars = []

    print("\nChoose whether each controllable variable is fixed or mutable in the search.")
    for col in controllable_cols:
        is_mutable = ask_yes_no(f"Allow '{col}' to vary?", default="N")
        if is_mutable:
            mutable_vars.append(col)
        else:
            fixed_vars.append(col)

    return fixed_vars, mutable_vars

def generate_full_combination_candidates(
    df: pd.DataFrame,
    predictor_cols: list[str],
    baseline_values: dict,
    controllable_cols: list[str],
    fixed_vars: list[str],
    mutable_vars: list[str],
    numeric_strategy: str = NUMERIC_POOL_STRATEGY_DEFAULT,
    max_unique_numeric_values: int = MAX_UNIQUE_NUMERIC_VALUES_PER_VARIABLE,
    nearest_k: int = NUMERIC_NEAREST_VALUES_DEFAULT,
    max_full_rows: int = MAX_FULL_COMBINATION_ROWS,
):
    """
    Generate full Cartesian-product combinations over mutable variables.
    Uses only realistic values observed in the dataset.
    Does NOT retrain any model.
    """
    if len(mutable_vars) == 0:
        one_row = pd.DataFrame([baseline_values], columns=predictor_cols)
        return one_row, {}, 1

    candidate_pools = {}
    for col in mutable_vars:
        pool = build_candidate_pool_for_variable(
            df=df,
            col=col,
            baseline_value=baseline_values.get(col, None),
            max_unique_numeric_values=max_unique_numeric_values,
            numeric_strategy=numeric_strategy,
            nearest_k=nearest_k,
        )

        if len(pool) == 0:
            raise ValueError(f"No candidate pool could be built for mutable variable '{col}'.")

        candidate_pools[col] = pool

    # Estimate total combinations before generation
    total_combinations = 1
    for col in mutable_vars:
        total_combinations *= len(candidate_pools[col])

    return None, candidate_pools, total_combinations

def run_full_combination_search(
    df: pd.DataFrame,
    artifacts: dict[str, TrainingArtifact],
    predictor_cols: list[str],
    target_cols: list[str],
    output_dir: Path,
):
    """
    Full-combination constrained optimization / exhaustive search for recommendation.
    Uses already trained artifact.final_pipeline.
    """
    available_targets = [t for t in target_cols if t in artifacts]
    if len(available_targets) == 0:
        raise ValueError("No trained targets available for full combination search.")

    print("\nAvailable trained targets:")
    for t in available_targets:
        print(f"- {t}")

    target = ask_text("Enter target to optimize", default=available_targets[0])
    if target not in artifacts:
        raise ValueError(f"Target '{target}' is not available in trained artifacts.")

    artifact = artifacts[target]
    model = artifact.final_pipeline

    # Resolve the 6 controllable variables against actual dataset columns
    requested_friendly = [
        "Experiment Time",
        "Pollutant Class",
        "Metal Atom",
        "Support Material Class",
        "Metal Loading",
        "Oxidant",
    ]

    controllable_cols = []
    for friendly in requested_friendly:
        resolved = resolve_searchable_variable_name(df, friendly)
        if resolved is not None and resolved in predictor_cols:
            controllable_cols.append(resolved)

    if len(controllable_cols) == 0:
        raise ValueError("None of the controllable variables were found in the current dataset/predictor set.")

    print("\nControllable variables found:")
    for col in controllable_cols:
        print(f"- {col}")

    baseline_values, original_reported_target = prompt_for_baseline_experiment(
        df=df,
        predictor_cols=predictor_cols,
        controllable_cols=controllable_cols,
    )

    fixed_vars, mutable_vars = prompt_for_fixed_and_mutable_variables(controllable_cols)

    print("\nNumeric candidate pool strategy options:")
    print("- unique   : use observed unique values, capped")
    print("- quantile : use quantile-subsampled observed values")
    print("- nearest  : use observed values nearest to baseline")
    numeric_strategy = ask_text(
        "Numeric pool strategy",
        default=NUMERIC_POOL_STRATEGY_DEFAULT
    ).strip().lower()

    if numeric_strategy not in {"unique", "quantile", "nearest"}:
        print(f"Unknown strategy '{numeric_strategy}', falling back to '{NUMERIC_POOL_STRATEGY_DEFAULT}'.")
        numeric_strategy = NUMERIC_POOL_STRATEGY_DEFAULT

    # Estimate search space before building all rows
    _, candidate_pools, total_combinations = generate_full_combination_candidates(
        df=df,
        predictor_cols=predictor_cols,
        baseline_values=baseline_values,
        controllable_cols=controllable_cols,
        fixed_vars=fixed_vars,
        mutable_vars=mutable_vars,
        numeric_strategy=numeric_strategy,
        max_unique_numeric_values=MAX_UNIQUE_NUMERIC_VALUES_PER_VARIABLE,
        nearest_k=NUMERIC_NEAREST_VALUES_DEFAULT,
        max_full_rows=MAX_FULL_COMBINATION_ROWS,
    )

    print("\nCandidate pool sizes:")
    for col in mutable_vars:
        print(f"- {col}: {len(candidate_pools[col])} values")

    print(f"\nExpected number of combinations: {total_combinations:,}")

    if total_combinations > MAX_FULL_COMBINATION_ROWS:
        print(f"Warning: this exceeds the soft limit of {MAX_FULL_COMBINATION_ROWS:,} combinations.")
        continue_anyway = ask_yes_no("Continue?", default="N")
        if not continue_anyway:
            print("Search cancelled.")
            return pd.DataFrame()

    # Optional random sampling if very large
    sample_after_generation = False
    sample_size = MAX_FULL_COMBINATION_ROWS
    if total_combinations > MAX_FULL_COMBINATION_ROWS and RANDOM_SAMPLE_IF_EXCEEDS:
        sample_after_generation = ask_yes_no(
            f"Randomly sample down to about {MAX_FULL_COMBINATION_ROWS:,} candidates after generation?",
            default="Y"
        )

    # Build full Cartesian product
    if len(mutable_vars) == 0:
        candidate_rows = [baseline_values.copy()]
    else:
        pool_names = list(candidate_pools.keys())
        pool_lists = [candidate_pools[c] for c in pool_names]

        candidate_rows = []
        for combo in product(*pool_lists):
            row = baseline_values.copy()

            # fixed variables stay baseline
            for col in fixed_vars:
                row[col] = baseline_values[col]

            # mutable variables take combo values
            for col, val in zip(pool_names, combo):
                row[col] = val

            # any missing predictor still gets safe default
            for col in predictor_cols:
                if col not in row or row[col] is None or (isinstance(row[col], float) and pd.isna(row[col])):
                    row[col] = get_default_value_for_column(df, col)

            candidate_rows.append(row)

    candidates_df = pd.DataFrame(candidate_rows)

    # Keep only predictor columns in the same order as model training
    for col in predictor_cols:
        if col not in candidates_df.columns:
            candidates_df[col] = get_default_value_for_column(df, col)

    candidates_df = candidates_df[predictor_cols].copy().drop_duplicates().reset_index(drop=True)

    if len(candidates_df) == 0:
        raise ValueError("No candidate rows were generated.")

    if sample_after_generation and len(candidates_df) > sample_size:
        candidates_df = candidates_df.sample(sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Randomly sampled candidates down to {len(candidates_df):,} rows.")

    # Predict baseline
    baseline_df = pd.DataFrame([baseline_values])
    for col in predictor_cols:
        if col not in baseline_df.columns:
            baseline_df[col] = get_default_value_for_column(df, col)
    baseline_df = baseline_df[predictor_cols]

    baseline_pred = float(model.predict(baseline_df)[0])

    # Predict candidates
    candidate_preds = model.predict(candidates_df)

    result_df = candidates_df.copy()

    # Add baseline + comparison columns for controllable variables
    for col in controllable_cols:
        result_df[f"Baseline__{col}"] = baseline_values.get(col, None)
        result_df[f"Candidate__{col}"] = result_df[col]

    result_df["BaselinePredictedTarget"] = baseline_pred
    result_df["CandidatePredictedTarget"] = candidate_preds
    result_df["PredictedImprovement"] = result_df["CandidatePredictedTarget"] - result_df["BaselinePredictedTarget"]
    result_df["OriginalReportedTarget"] = original_reported_target if original_reported_target != "" else np.nan

    changed_vars = []
    changed_details = []
    for _, row in result_df.iterrows():
        names, details = format_changed_variables(
            baseline_row=baseline_values,
            candidate_row=row.to_dict(),
            compare_cols=controllable_cols,
        )
        changed_vars.append(names)
        changed_details.append(details)

    result_df["ChangedVariables"] = changed_vars
    result_df["ChangedDetails"] = changed_details

    # Add some top-level context columns
    result_df.insert(0, "Target", target)
    result_df.insert(1, "BestModel", artifact.best_model_name)

    # Rank descending by predicted target
    result_df = result_df.sort_values("CandidatePredictedTarget", ascending=False).reset_index(drop=True)

    top_n = ask_int("How many top combinations to save", DEFAULT_TOP_N_SEARCH_RESULTS)
    top_df = result_df.head(top_n).copy()

    out_file = output_dir / "full_combination_constrained_search.csv"
    top_df.to_csv(out_file, index=False)

    # Console summary
    print("\nFull-combination constrained search summary")
    print(f"- Generated candidate rows: {len(result_df):,}")
    print(f"- Baseline predicted target: {baseline_pred:.6f}")

    best_row = top_df.iloc[0]
    print(f"- Best predicted target: {best_row['CandidatePredictedTarget']:.6f}")
    print(f"- Predicted improvement: {best_row['PredictedImprovement']:.6f}")
    print(f"- Variables changed in best candidate: {best_row['ChangedVariables']}")
    print(f"- Saved top {top_n} results to: {out_file}")

    print("\nTop results preview:")
    preview_cols = [
        "CandidatePredictedTarget",
        "PredictedImprovement",
        "ChangedVariables",
        "ChangedDetails",
    ]
    preview_cols = [c for c in preview_cols if c in top_df.columns]
    print(top_df[preview_cols].head(min(top_n, 10)))

    return top_df

# =============================================================================
# TRAINING CACHE HELPERS
# =============================================================================

def get_model_cache_dir() -> Path:
    """
    Stable cache directory for trained artifacts across runs.
    Do not use the timestamped run output folder here.
    """
    cache_dir = Path(BASE_OUTPUT_DIR) / "model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def save_training_bundle(cache_dir: Path, benchmark_df, best_df, imp_df, artifacts):
    cache_dir = ensure_dir(cache_dir)
    benchmark_df.to_csv(cache_dir / "model_benchmark.csv", index=False)
    best_df.to_csv(cache_dir / "best_model_per_target.csv", index=False)
    imp_df.to_csv(cache_dir / "feature_importance_summary.csv", index=False)
    joblib.dump(artifacts, cache_dir / "artifacts.joblib")
    print(f"[INFO] Saved trained models to cache: {cache_dir}")

def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    
def load_training_bundle(cache_dir: Path):
    artifacts_path = cache_dir / "artifacts.joblib"
    benchmark_path = cache_dir / "model_benchmark.csv"
    best_path = cache_dir / "best_model_per_target.csv"
    imp_path = cache_dir / "feature_importance_summary.csv"

    if not artifacts_path.exists():
        raise FileNotFoundError(f"No saved model cache found at: {artifacts_path}")

    benchmark_df = _safe_read_csv(benchmark_path)
    best_df = _safe_read_csv(best_path)
    imp_df = _safe_read_csv(imp_path)
    artifacts = joblib.load(artifacts_path)

    print(f"[INFO] Loaded trained models from cache: {cache_dir}")
    return benchmark_df, best_df, imp_df, artifacts


def get_or_train_models(df, predictor_cols, target_cols, output_dir):
    """
    Ask once whether to retrain models or reuse cached models.
    """
    cache_dir = get_model_cache_dir()
    cache_exists = (cache_dir / "artifacts.joblib").exists()

    if cache_exists:
        retrain = ask_yes_no("Saved trained models found. Retrain models", default="N")
        if not retrain:
            benchmark_df, best_df, imp_df, artifacts = load_training_bundle(cache_dir)

            # optional: copy cached summary tables into this run folder
            if benchmark_df is not None and not benchmark_df.empty:
                benchmark_df.to_csv(cache_dir / "model_benchmark.csv", index=False)

            if best_df is not None and not best_df.empty:
                best_df.to_csv(cache_dir / "best_model_per_target.csv", index=False)

            if imp_df is not None and not imp_df.empty:
                imp_df.to_csv(cache_dir / "feature_importance_summary.csv", index=False)

            return benchmark_df, best_df, imp_df, artifacts
    else:
        print("[INFO] No saved trained model cache found. Training is required.")

    print("[INFO] Training and benchmarking models...")
    benchmark_df, best_df, imp_df, artifacts = benchmark_all_targets(
        df, predictor_cols, target_cols, output_dir
    )
    save_training_bundle(cache_dir, benchmark_df, best_df, imp_df, artifacts)
    return benchmark_df, best_df, imp_df, artifacts

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 88)
    print("THESIS-GRADE DOI-AWARE ML PIPELINE FOR SINGLE-ATOM CATALYST DATASETS")
    print("=" * 88)
    print("This version uses DOI-grouped validation so rows from the same paper")
    print("do not leak into both training and testing.")
    print()

    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(
            f"Could not find '{DATA_FILE}'. Put the CSV in the same folder as this script "
            "or change DATA_FILE in the config section."
        )

    output_dir = create_output_dir(BASE_OUTPUT_DIR)
    df = load_and_clean_data(DATA_FILE)

    schema_report = validate_schema(df)
    if schema_report["missing_required"]:
        print("Schema issues found:")
        for item in schema_report["missing_required"]:
            print(f"- {item}")
        raise ValueError("Please fix the dataset/schema issues above before running the pipeline.")

    predictor_cols = schema_report["active_predictors"]
    target_cols = schema_report["active_targets"]

    print(f"Rows: {len(df)}")
    print(f"Columns: {df.shape[1]}")
    print(f"Unique DOIs: {df[SCHEMA['group_col']].nunique(dropna=True)}")
    print(f"Predictors used: {len(predictor_cols)}")
    print(f"Targets found: {len(target_cols)}")

    save_schema_and_run_info(df, predictor_cols, target_cols, output_dir)
    save_target_coverage(df, target_cols, output_dir)

    print("\nAvailable actions")
    print("1. Run full analysis")
    print("2. Run EDA only")
    print("3. Run model training only")
    print("4. Screen candidates")
    print("5. Predict one custom condition")
    print("6. Full combination constrained search")
    print("7. Exit")

    choice = ask_text("Choose an option", default="1")

    artifacts = {}
    best_df = pd.DataFrame()

    if choice == "1":
        print("\nRunning EDA...")
        run_eda(df, output_dir, predictor_cols, target_cols)
        print("EDA saved.")

        print("\nPreparing models...")
        benchmark_df, best_df, imp_df, artifacts = get_or_train_models(df, predictor_cols, target_cols, output_dir)
        if benchmark_df.empty:
            print("No benchmark results were produced. Check target coverage and DOI counts.")
        else:
            print("\nBest model per target:")
            print(best_df)

    elif choice == "2":
        print("\nRunning EDA...")
        run_eda(df, output_dir, predictor_cols, target_cols)
        print("EDA saved.")

    elif choice == "3":
        print("\nPreparing models...")
        benchmark_df, best_df, imp_df, artifacts = get_or_train_models(
            df, predictor_cols, target_cols, output_dir
        )
        if benchmark_df.empty:
            print("No benchmark results were produced. Check target coverage and DOI counts.")
        else:
            print("\nBest model per target:")
            print(best_df)

    elif choice == "4":
        print("\nTraining models first because candidate screening needs a fitted best model...")
        benchmark_df, best_df, imp_df, artifacts = get_or_train_models(
            df, predictor_cols, target_cols, output_dir
        )
        if best_df.empty:
            raise ValueError("No models were successfully trained.")
        print("\nAvailable targets:")
        for t in target_cols:
            if t in artifacts:
                print(f"- {t}")
        target = ask_text("Enter target to screen", default=target_cols[0])
        use_filter = ask_yes_no("Filter screening by pollutant class?", default="N")
        pollutant_class = None
        if use_filter:
            pollutant_class = ask_text("Enter pollutant class exactly as in dataset")
        top_n = ask_int("How many top candidates", 20)
        screened = screen_candidates(
            df=df,
            artifacts=artifacts,
            predictor_cols=predictor_cols,
            target=target,
            output_dir=output_dir,
            top_n=top_n,
            pollutant_class_filter=pollutant_class,
            file_name="candidate_screening.csv" if pollutant_class is None else "candidate_screening_by_pollutant_class.csv",
        )
        print("\nTop screened candidates:")
        print(screened.head(min(top_n, 10)))

    elif choice == "5":
        print("\nPreparing models because custom prediction needs a fitted best model...")
        benchmark_df, best_df, imp_df, artifacts = get_or_train_models(
            df, predictor_cols, target_cols, output_dir
        )
        if best_df.empty:
            raise ValueError("No models were successfully trained.")
        print("\nAvailable targets:")
        for t in target_cols:
            if t in artifacts:
                print(f"- {t}")
        target = ask_text("Enter target to predict", default=target_cols[0])
        result = prompt_for_single_prediction(df, artifacts, predictor_cols, target, output_dir)
        print("\nPrediction result:")
        print(result)

    elif choice == "6":
        print("\nPreparing models because full combination constrained search needs a fitted best model...")
        benchmark_df, best_df, imp_df, artifacts = get_or_train_models(
            df, predictor_cols, target_cols, output_dir
        )
        if best_df.empty:
            raise ValueError("No models were successfully trained.")
        run_full_combination_search(
            df=df,
            artifacts=artifacts,
            predictor_cols=predictor_cols,
            target_cols=target_cols,
            output_dir=output_dir,
        )

    else:
        print("Exiting.")
        return

    print(f"\nDone. Outputs saved in:\n{output_dir.resolve()}")


if __name__ == "__main__":
    main()
