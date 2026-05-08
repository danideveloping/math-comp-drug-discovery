import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
os.environ["TABPFN_TOKEN"] ="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiZDk2ZjAyNzEtODI3MC00MGIxLWEwN2QtYzQ1MzBlNzlmMGY3IiwiZXhwIjoxODA5NzgxNjc1fQ.FC-Rkp9LQTru55xHAZ8G-NYIHI4Q0HZ_jMuzV9GjI4M"

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/AdrientheFragrance/"
    "Advanced-PCOS-Prediction-Model/main/PCOS_infertility.csv"
)


@dataclass
class EvalResult:
    model: str
    roc_auc: float
    pr_auc: float
    f1: float
    recall: float
    accuracy: float
    seconds: float


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def find_target_column(df: pd.DataFrame) -> str:
    candidates = [
        "pcos_y_n",
        "pcos",
        "target",
        "label",
        "diagnosis",
    ]
    for name in candidates:
        if name in df.columns:
            return name

    fuzzy = [c for c in df.columns if "pcos" in c]
    if fuzzy:
        return fuzzy[0]
    raise ValueError("Could not find target column. Please pass --target-col.")


def make_binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0).astype(int)

    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"1": 1, "0": 0, "yes": 1, "no": 0, "y": 1, "n": 0, "true": 1, "false": 0})
    )
    if mapped.isna().any():
        # Fallback: treat uncommon values as category codes, then binarize
        codes = series.astype("category").cat.codes
        return (codes > 0).astype(int)
    return mapped.astype(int)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def get_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("clf", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced")),
            ]
        ),
    }

    try:
        from tabpfn import TabPFNClassifier  # type: ignore

        if not os.environ.get("TABPFN_TOKEN"):
            print(
                "\nTABPFN_TOKEN is not set. "
                "If license/login prompts fail, set it first, e.g.:\n"
                "  export TABPFN_TOKEN='your_token'   (Linux/macOS)\n"
                "  set TABPFN_TOKEN=your_token        (Windows cmd)\n"
                "  $env:TABPFN_TOKEN='your_token'     (PowerShell)"
            )

        # TabPFN can consume mixed numeric arrays; we still preprocess to handle missing/categorical robustly.
        models["TabPFN"] = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("clf", TabPFNClassifier()),
            ]
        )
    except Exception:
        pass

    return models


def evaluate_model(
    name: str, model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold
) -> EvalResult:
    start = time.perf_counter()
    y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    elapsed = time.perf_counter() - start

    return EvalResult(
        model=name,
        roc_auc=roc_auc_score(y, y_prob),
        pr_auc=average_precision_score(y, y_prob),
        f1=f1_score(y, y_pred),
        recall=recall_score(y, y_pred),
        accuracy=accuracy_score(y, y_pred),
        seconds=elapsed,
    )


def print_results(results: List[EvalResult]) -> None:
    print("\n=== Benchmark Results (Stratified CV) ===")
    sorted_res = sorted(results, key=lambda r: r.roc_auc, reverse=True)
    for r in sorted_res:
        print(
            f"{r.model:18s} | ROC-AUC: {r.roc_auc:.4f} | PR-AUC: {r.pr_auc:.4f} | "
            f"F1: {r.f1:.4f} | Recall: {r.recall:.4f} | Acc: {r.accuracy:.4f} | "
            f"Time(s): {r.seconds:.2f}"
        )


def save_results(results: List[EvalResult], output_path: str) -> None:
    pd.DataFrame([r.__dict__ for r in results]).sort_values("roc_auc", ascending=False).to_csv(
        output_path, index=False
    )
    print(f"\nSaved results to: {output_path}")


def load_data(path_or_url: str) -> pd.DataFrame:
    print(f"Loading dataset from: {path_or_url}")
    return pd.read_csv(path_or_url)


def main() -> None:
    parser = argparse.ArgumentParser(description="PCOS benchmark using TabPFN and classic ML baselines.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATASET_URL, help="CSV file path or URL.")
    parser.add_argument("--target-col", type=str, default=None, help="Target column name (optional).")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument("--results-csv", type=str, default="pcos_benchmark_results.csv")
    # Colab/Jupyter injects extra CLI args (for example: -f kernel.json).
    # parse_known_args keeps our arguments and safely ignores unknown ones.
    args, _unknown = parser.parse_known_args()

    df = normalize_columns(load_data(args.data))
    target_col = args.target_col if args.target_col else find_target_column(df)
    print(f"Using target column: {target_col}")

    # Remove obvious ID-like columns if present.
    id_like_cols = [c for c in df.columns if "id" == c or c.endswith("_id") or "patient_file_no" in c]
    feature_df = df.drop(columns=[target_col] + id_like_cols, errors="ignore")
    y = make_binary_target(df[target_col])

    # Make object columns consistent and parse numeric-looking strings.
    for col in feature_df.columns:
        if feature_df[col].dtype == object:
            cleaned = feature_df[col].astype(str).str.strip()
            maybe_num = pd.to_numeric(cleaned, errors="coerce")
            # If most values are numeric-like, keep numeric; otherwise keep string category.
            if maybe_num.notna().mean() > 0.8:
                feature_df[col] = maybe_num
            else:
                feature_df[col] = cleaned.replace({"nan": np.nan, "": np.nan})

    print(f"Rows: {len(feature_df)}, Features: {feature_df.shape[1]}, Positive rate: {y.mean():.3f}")
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    preprocessor = build_preprocessor(feature_df)
    models = get_models(preprocessor)

    if "TabPFN" not in models:
        print(
            "\nTabPFN was not detected. Install it with:\n"
            "  pip install tabpfn\n"
            "Then rerun to include it in the benchmark."
        )

    results: List[EvalResult] = []
    for name, model in models.items():
        print(f"Running: {name}")
        results.append(evaluate_model(name, model, feature_df, y, cv))

    print_results(results)
    save_results(results, args.results_csv)


if __name__ == "__main__":
    main()
