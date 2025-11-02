# src/run_benchmark_csv.py
# CSV-only end-to-end: build a pediatric appointments table with a no-show label,
# then train Logistic, Elastic Net, Random Forest, and (optionally) XGBoost.

from __future__ import annotations
import os, math, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Try to import XGBoost; skip it if not installed
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# -------------------------
# Config
# -------------------------
SEED = 42
TARGET_NOSHOW_RATE = 0.20          # overall no-show base rate to simulate
LEAD_TIME_MIN, LEAD_TIME_MAX = 1, 30   # days between scheduling and appointment
CLINIC_HOURS_START, CLINIC_HOURS_END = 8, 17  # [8, 16] inclusive

# Project paths (relative to repo root)
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "synthea_csv"
PROC_DIR = ROOT / "data" / "processed" / "appointments_ml"
METRICS_DIR = ROOT / "models" / "metrics"
PROC_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# Short, user-editable blurbs rendered under the overview charts
MODEL_DESCRIPTIONS = {
    "logistic_regression": "This model attempts to model input features and a target varibale using a logistic function. It takes in the features, gives them a weight, and transforms the weighted sum through a sigmoid curve to produce probabilities between 0 and 1. The goal of this model is to minimize the Log Loss. This model performs poorly with multicolinearity within the features.",
    "elastic_net_logistic": "This model is improved at handling collinearity and overfitting. It combines regularization factors from both Ridge and Lasso models, used to both stablize the model and select important features.",
    "random_forest": "This model takes decision trees that where independently trained on a bootstrap sample of the data and uses random subsets at each split, then averages the results of all the trees (that is the final result). This model handles non-linearity quite well as it is not designed to fit a straight line.",
    "xgboost": "This model is a gradient boosting algorithm uses decision trees that are built sequentially. This way, they are able to correct eachothers previous mistakes. This model is built for speed and regulariztion, specifically good for larger datasets."
}

# -------------------------
# Helpers
# -------------------------
def read_csv(name: str) -> pd.DataFrame:
    fp = RAW_DIR / name
    if not fp.exists():
        raise FileNotFoundError(f"Expected {fp} but not found.")
    return pd.read_csv(fp)

def to_dt(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)

def evaluate_model(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    pipeline.fit(X_train, y_train)
    prob = pipeline.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "model": name,
        "auc_roc": float(roc_auc_score(y_test, prob)),
        "auprc": float(average_precision_score(y_test, prob)),
        "accuracy@0.5": float(accuracy_score(y_test, pred)),
        "precision@0.5": float(precision_score(y_test, pred, zero_division=0)),
        "recall@0.5": float(recall_score(y_test, pred, zero_division=0)),
        "f1@0.5": float(f1_score(y_test, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_test, prob)),
        "expected_noshow_pred": float(prob.mean()),
        "observed_noshow_test": float(y_test.mean())
    }
    print(f"[{name}] AUC={metrics['auc_roc']:.3f}  AUPRC={metrics['auprc']:.3f}  "
          f"Acc@0.5={metrics['accuracy@0.5']:.3f}  ExpNoShow={metrics['expected_noshow_pred']:.3f}")
    return metrics

# -------------------------
# 1) Load required CSVs
# -------------------------
print(f"Reading CSVs from: {RAW_DIR}")
patients = read_csv("patients.csv")
encounters = read_csv("encounters.csv")

# Column hygiene
if "Id" not in patients.columns:
    # try to recover if someone renamed it
    cand = [c for c in patients.columns if c.lower() in ("id", "patient", "patient_id")]
    if not cand:
        raise ValueError("patients.csv must contain an 'Id' column.")
    patients = patients.rename(columns={cand[0]: "Id"})

if "BIRTHDATE" not in patients.columns:
    raise ValueError("patients.csv missing 'BIRTHDATE'.")

for col in ("PATIENT", "START"):
    if col not in encounters.columns:
        raise ValueError(f"encounters.csv missing '{col}'.")

patients["BIRTHDATE"] = to_dt(patients["BIRTHDATE"])
encounters["START"] = to_dt(encounters["START"])
encounters = encounters.dropna(subset=["START"]).copy()

# Attach demographics to encounters & compute age at encounter; keep pediatric
enc_demo = encounters.merge(
    patients[["Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"]],
    left_on="PATIENT", right_on="Id", how="left"
)
enc_demo["age_at_enc"] = (enc_demo["START"] - enc_demo["BIRTHDATE"]).dt.days / 365.25
enc_demo = enc_demo[(enc_demo["age_at_enc"] <= 18 + 1e-9)].copy()

if enc_demo.empty:
    # If no pediatric encounters, just proceed with all (rare if you generated 0–18)
    enc_demo = encounters.merge(
        patients[["Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"]],
        left_on="PATIENT", right_on="Id", how="left"
    )
    enc_demo["age_at_enc"] = (enc_demo["START"] - enc_demo["BIRTHDATE"]).dt.days / 365.25

# -------------------------
# 2) Build synthetic appointments with no-shows
# -------------------------
print("Synthesizing appointments (shows + no-shows)…")

# a) SHOW appointments: one per encounter, scheduled some days earlier
shows = enc_demo[["PATIENT", "START"]].copy()
lead = rng.integers(LEAD_TIME_MIN, LEAD_TIME_MAX + 1, size=len(shows))
shows["scheduled_dt"] = shows["START"] - pd.to_timedelta(lead, unit="D")
shows["scheduled_hour"] = rng.integers(CLINIC_HOURS_START, CLINIC_HOURS_END, size=len(shows))
shows["show"] = 1

n_shows = len(shows)
if n_shows == 0:
    raise RuntimeError("No encounters found to seed show appointments.")

# b) NOSHOW appointments: add extra scheduled rows without an encounter
#    to reach TARGET_NOSHOW_RATE on average.
n_total = math.ceil(n_shows / (1.0 - TARGET_NOSHOW_RATE))
n_noshows = max(1, n_total - n_shows)

print(
    f"Counts — from Synthea (shows)={n_shows}, synthetic_noshows={n_noshows}, total={n_total}"
)

patient_ids = enc_demo["PATIENT"].dropna().unique()
if len(patient_ids) == 0:
    raise RuntimeError("No pediatric patients found in encounters; cannot synthesize no-shows.")

start_min, start_max = enc_demo["START"].min(), enc_demo["START"].max()
if pd.isna(start_min) or pd.isna(start_max):
    start_min = pd.Timestamp("2023-01-01", tz="UTC")
    start_max = pd.Timestamp("2024-01-01", tz="UTC")

rand_seconds = rng.integers(int(start_min.value / 1e9), int(start_max.value / 1e9), size=n_noshows)
noshows = pd.DataFrame({
    "PATIENT": rng.choice(patient_ids, size=n_noshows, replace=True),
    "scheduled_dt": pd.to_datetime(rand_seconds, unit="s", utc=True),
    "scheduled_hour": rng.integers(CLINIC_HOURS_START, CLINIC_HOURS_END, size=n_noshows),
    "show": 0,
    "START": pd.NaT,   # no encounter occurred
})

appointments = pd.concat(
    [
        shows[["PATIENT", "scheduled_dt", "scheduled_hour", "show", "START"]],
        noshows[["PATIENT", "scheduled_dt", "scheduled_hour", "show", "START"]],
    ],
    ignore_index=True
)

# Join patient demographics
appointments = appointments.merge(
    patients[["Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"]],
    left_on="PATIENT", right_on="Id", how="left"
)

# Core features
appointments["scheduled_dt"] = pd.to_datetime(appointments["scheduled_dt"], errors="coerce", utc=True)
appointments["START"] = pd.to_datetime(appointments["START"], errors="coerce", utc=True)
appointments["BIRTHDATE"] = pd.to_datetime(appointments["BIRTHDATE"], errors="coerce", utc=True)
appointments["age_years"] = (appointments["scheduled_dt"] - appointments["BIRTHDATE"]).dt.days / 365.25
appointments["weekday"] = appointments["scheduled_dt"].dt.weekday  # 0=Mon
appointments["month"] = appointments["scheduled_dt"].dt.month
appointments["lead_time_days"] = (appointments["START"] - appointments["scheduled_dt"]).dt.days
missing_lead = appointments["lead_time_days"].isna()
appointments.loc[missing_lead, "lead_time_days"] = rng.integers(LEAD_TIME_MIN, LEAD_TIME_MAX + 1, size=missing_lead.sum())

# Prior encounters in windows before scheduled date
enc_min = enc_demo[["PATIENT", "START"]].rename(columns={"START": "enc_start"}).copy()

def count_prior(df_apt, df_enc, days: int) -> pd.Series:
    # Expand by patient to compare each appointment with all their encounters
    m = df_apt[["PATIENT", "scheduled_dt"]].merge(df_enc, on="PATIENT", how="left")
    m["cutoff"] = m["scheduled_dt"] - pd.to_timedelta(days, unit="D")
    m["is_prior"] = (m["enc_start"] < m["scheduled_dt"]) & (m["enc_start"] >= m["cutoff"])
    # Count back to the original appointment rows via the merge's left index
    counts = m.groupby(m.index)["is_prior"].sum().astype(int)
    return counts.reindex(df_apt.index, fill_value=0)

appointments["prior_enc_30"]  = count_prior(appointments, enc_min, 30)
appointments["prior_enc_180"] = count_prior(appointments, enc_min, 180)
appointments["prior_enc_365"] = count_prior(appointments, enc_min, 365)
appointments["is_new_patient"] = (appointments["prior_enc_365"] == 0).astype(int)

# Prior no-shows per patient (strictly before current appointment)
appointments = appointments.sort_values(["PATIENT", "scheduled_dt"]).copy()
_noshow_flag = (appointments["show"] == 0).astype(int)
appointments["prior_noshows"] = (
    _noshow_flag.groupby(appointments["PATIENT"]).cumsum().shift(fill_value=0).astype(int)
)

# Keep only pediatric ages (defensive)
appointments = appointments[appointments["age_years"].between(0, 18, inclusive="both")].copy()
print(
    "Post-filter pediatrics — shows=", int((appointments["show"] == 1).sum()),
    ", noshows=", int((appointments["show"] == 0).sum()),
    ", total=", len(appointments),
    ", no_show_rate=", float(1 - appointments["show"].mean())
)

# Final selected columns
keep = [
    "PATIENT", "scheduled_dt", "scheduled_hour", "weekday", "month",
    "lead_time_days", "age_years", "GENDER", "RACE", "ETHNICITY",
    "prior_enc_30", "prior_enc_180", "prior_enc_365", "is_new_patient", "show"
]
# Insert the new column into the exported dataset
keep.insert(keep.index("prior_enc_30"), "prior_noshows")
appointments = appointments[keep].reset_index(drop=True)

# Save the modeling dataset
out_csv = PROC_DIR / "appointments_ml.csv"
appointments.to_csv(out_csv, index=False)
print(f"Saved modeling dataset: {out_csv}  (rows={len(appointments)}, no_show_rate={1 - (appointments['show'].mean()):.3f})")

# -------------------------
# 3) Train/test & preprocessing
# -------------------------
X = appointments.drop(columns=["show", "scheduled_dt"])
y = appointments["show"].astype(int)

numeric_features = [
    "age_years", "lead_time_days", "prior_noshows", "prior_enc_30", "prior_enc_180",
    "prior_enc_365", "is_new_patient", "scheduled_hour", "weekday", "month"
]
categorical_features = ["GENDER", "RACE", "ETHNICITY"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocess = ColumnTransformer([
    ("num", num_pipe, numeric_features),
    ("cat", cat_pipe, categorical_features),
])

# -------------------------
# 4) Train four models
# -------------------------
results = []

# Logistic Regression (L2)
logreg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=SEED)),
])
results.append(evaluate_model("logistic_regression", logreg, X_train, X_test, y_train, y_test))

# Elastic Net Logistic
elastic = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(
        penalty="elasticnet", l1_ratio=0.5, solver="saga",
        max_iter=5000, class_weight="balanced", random_state=SEED)),
])
results.append(evaluate_model("elastic_net_logistic", elastic, X_train, X_test, y_train, y_test))

# Random Forest
rf = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
])
results.append(evaluate_model("random_forest", rf, X_train, X_test, y_train, y_test))

# XGBoost (optional)
if HAS_XGB:
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = float(neg / max(1, pos))
    xgb = Pipeline([
        ("prep", preprocess),
        ("clf", XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, eval_metric="logloss",
            tree_method="hist", scale_pos_weight=spw)),
    ])
    results.append(evaluate_model("xgboost", xgb, X_train, X_test, y_train, y_test))
else:
    print("xgboost not installed; skipping XGBoost.")

# -------------------------
# 5) Save metrics table
# -------------------------
metrics_df = pd.DataFrame(results)
metrics_csv = METRICS_DIR / "metrics.csv"
if metrics_csv.exists():
    old = pd.read_csv(metrics_csv)
    metrics_df = pd.concat([old, metrics_df], ignore_index=True)
metrics_df.to_csv(metrics_csv, index=False)
print(f"Saved metrics to: {metrics_csv}")

# -------------------------
# 6) Define pipelines dict for downstream reporting
# -------------------------

name_to_pipeline = {
    "logistic_regression": logreg,
    "elastic_net_logistic": elastic,
    "random_forest": rf,
}
if HAS_XGB:
    name_to_pipeline["xgboost"] = xgb

# -------------------------
# 7) Model overview charts: ROC AUC and Brier Score with descriptions
# -------------------------
try:
    all_metrics = pd.read_csv(METRICS_DIR / "metrics.csv")
    if all_metrics.empty:
        raise RuntimeError("metrics.csv is empty")

    order = [m for m in ["logistic_regression", "elastic_net_logistic", "random_forest", "xgboost"]
             if m in set(all_metrics["model"]) ]

    # Combined overview: both ROC AUC and Brier Score on one page
    import matplotlib.gridspec as gridspec

    # Helper to format model names: remove underscores, title case
    def format_model_name(name: str) -> str:
        return name.replace("_", " ").title()

    # Prepare data for both metrics
    plot_dfs = {}
    for metric in ["auc_roc", "brier"]:
        df = all_metrics.groupby("model", as_index=False)[metric].mean()
        if order:
            df["order"] = df["model"].map({m: i for i, m in enumerate(order)})
            df = df.sort_values("order")
        plot_dfs[metric] = df

    # Create figure with 2 bar charts + descriptions panel (even smaller)
    fig = plt.figure(figsize=(3.5, 3))
    gs = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[1.5, 1.5, 1.0], hspace=0.6)

    # ROC AUC chart
    ax1 = fig.add_subplot(gs[0, 0])
    df1 = plot_dfs["auc_roc"]
    model_labels = [format_model_name(m) for m in df1["model"]]
    ax1.bar(model_labels, df1["auc_roc"], color="black", width=0.6)
    ax1.set_title("ROC AUC (Receiver Operating Characteristic Area Under Curve)", fontsize=7, fontweight="bold")
    ax1.set_ylabel("ROC AUC", fontsize=9)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticklabels(model_labels, rotation=0, ha="center", fontsize=4)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Brier Score chart (lower is better)
    ax2 = fig.add_subplot(gs[1, 0])
    df2 = plot_dfs["brier"]
    model_labels2 = [format_model_name(m) for m in df2["model"]]
    ax2.bar(model_labels2, df2["brier"], color="black", width=0.6)
    ax2.set_title("Brier Score (lower is better)", fontsize=7, fontweight="bold")
    ax2.set_ylabel("Brier Score", fontsize=9)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xticklabels(model_labels2, rotation=0, ha="center", fontsize=4)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # Descriptions panel (edit MODEL_DESCRIPTIONS at top of this file)
    ax_txt = fig.add_subplot(gs[2, 0])
    ax_txt.axis("off")
    lines = []
    for m in df1["model"].tolist():
        desc = MODEL_DESCRIPTIONS.get(m, "")
        formatted_name = format_model_name(m)
        if desc:
            lines.append(f"{formatted_name}: {desc}")
        else:
            lines.append(f"{formatted_name}: (description not provided)")
    ax_txt.text(0.00, 0.95, "\n\n".join(lines), va="top", ha="left", fontsize=4,
                wrap=True, transform=ax_txt.transAxes)

    fig.tight_layout()
    # Expand the descriptions axis to use nearly full figure width, independent of
    # the left margin required by the y-axis labels on the bar charts above.
    try:
        pos = ax_txt.get_position()
        ax_txt.set_position([0.01, pos.y0, 0.98, pos.height])
    except Exception:
        pass
    out_path = METRICS_DIR / "overview_auc_roc_auprc.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined overview chart to: {out_path}")
except Exception as e:
    print(f"Skipping model overview charts: {e}")

# -------------------------
# 8) Export simple test-set predictions (patient_id, no_show_percent) for XGBoost
# -------------------------
try:
    if HAS_XGB:
        # xgb pipeline was fit inside evaluate_model and remains fitted here
        if 'xgb' in locals():
            patient_ids = X_test["PATIENT"].reset_index(drop=True)
            probs = xgb.predict_proba(X_test)[:, 1]
            export_df = pd.DataFrame({
                "patient_id": patient_ids,
                "no_show_percent": np.round(probs * 100.0, 2),
            })
            export_path = METRICS_DIR / "test_patient_no_show_percent.csv"
            export_df.to_csv(export_path, index=False)
            print(f"Saved test predictions to: {export_path}")
        else:
            print("XGBoost pipeline not available for prediction; skipped export.")
    else:
        print("xgboost not installed; skipped test predictions export.")
except Exception as e:
    print(f"Failed exporting test predictions: {e}")

# -------------------------
# 9) Plot top patients by predicted no-show percentage
# -------------------------
try:
    pred_csv = METRICS_DIR / "test_patient_no_show_percent.csv"
    if pred_csv.exists():
        df_pred = pd.read_csv(pred_csv)
        if {"patient_id", "no_show_percent"}.issubset(df_pred.columns):
            # Aggregate per patient: use max predicted risk across their test appointments
            per_patient = (
                df_pred.groupby("patient_id", as_index=False)["no_show_percent"].max()
            )
            # Top N patients
            TOP_N = 10
            top = per_patient.sort_values("no_show_percent", ascending=False).head(TOP_N)

            # Prepare table data
            y_labels = top["patient_id"].astype(str).tolist()
            x_vals = top["no_show_percent"].tolist()
            table_rows = [[str(pid), f"{val:.1f}%"] for pid, val in zip(y_labels, x_vals)]
            table_data = [["Patient ID", "No - Show Percentage"]] + table_rows

            # Table-only figure
            fig_height = max(4.0, 0.6 * len(top) + 1.5)
            fig, ax_tbl = plt.subplots(figsize=(8, fig_height))
            ax_tbl.axis("off")
            tbl = ax_tbl.table(cellText=table_data, cellLoc="center", loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(11)
            tbl.scale(1.2, 1.5)

            # Header bold + light gray background
            for j in range(2):
                tbl[(0, j)].set_text_props(weight="bold")
                tbl[(0, j)].set_facecolor("#f0f0f0")

            # Color percentage cells light red when >= 75%
            for i, val in enumerate(x_vals, start=1):
                if val >= 75:
                    tbl[(i, 1)].set_facecolor("#ffcccc")

            # Title above table
            ax_tbl.set_title("Top 10 Patients by Predicted No-Show Percentage", fontsize=12, pad=12)

            fig.tight_layout()
            out_png = METRICS_DIR / "top_no_show_patients.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved table of top patients to: {out_png}")
        else:
            print("Prediction CSV missing required columns; skipping plot.")
    else:
        print("Prediction CSV not found; run script to export predictions first.")
except Exception as e:
    print(f"Failed plotting top patients: {e}")
