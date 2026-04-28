# %% [markdown]
# # Utilities

# %%
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr


# %%
from Utils.Config import ExperimentResult
from pprint import pprint


json_path = Path("Cache/MetaData/trial_0034/experiment_summary.json")

with open(json_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

exp = ExperimentResult.from_jsonable(raw)

pprint(exp)

# %% [markdown]
# # 0. Config

# %%
ROOT = Path("Cache/MetaData")  # trial folders root
SAVE_DIR = Path("analysis_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PERTURB_TO_FEATURE = {
    "localwarp": "shape",
    "patchshuffle": "shape",
    "patchrotation": "shape",
    "grayscale": "color",
    "bilateral": "texture",
    "bilateralfilter": "texture",
    "original": "clean",
    "none": "clean",
}


HPARAM_MAP = {
    # LocalWarp
    "alpha_localwarp": "alpha",
    "sigma_localwarp": "sigma",

    # Patch
    "grid_size": "grid_size",

    # Bilateral
    "bilateral_d": "bilateral_d",
    "sigma_color": "sigma_color",
    "sigma_space": "sigma_space",

    # Gaussian
    "gaussian_k": "gaussian_k",
    "gaussian_sigma": "gaussian_sigma",

    # Grayscale
    "gray_alpha": "gray_alpha",
}

HPARAM_COLS = [
    "alpha",
    "sigma",
    "grid_size",
    "bilateral_d",
    "sigma_color",
    "sigma_space",
    "gaussian_k",
    "gaussian_sigma",
    "gray_alpha",
]

PERTURB_HPARAM_COLS = {
    "localwarp": ["alpha", "sigma"],
    "bilateral": ["bilateral_d", "sigma_color", "sigma_space"],
    "patchshuffle": ["grid_size"],
    "patchrotation": ["grid_size"],
    "grayscale": ["gray_alpha"],
    "original": [],
}

ID_INTERNAL_DATASET = "imagenet"
ID_OOD_BASELINE_DATASET = "imagenet_200"
OOD_DATASET = "imagenet_r"


# %% [markdown]
# # 1. Utilities

# %%
def make_setting_id(row):
    perturb = row["perturbation"]
    cols = PERTURB_HPARAM_COLS.get(perturb, [])

    if not cols:
        return "default"

    parts = []
    for col in cols:
        val = row.get(col, np.nan)
        if pd.notna(val):
            parts.append(f"{col}={val}")

    return "|".join(parts) if parts else "default"

def normalize_perturbation(name):
    name = str(name).lower().replace("_", "").replace("-", "")

    if name in {"original", "clean", "none"}:
        return "original"
    if "localwarp" in name:
        return "localwarp"
    if "patchshuffle" in name:
        return "patchshuffle"
    if "patchrotation" in name or "patchrotate" in name:
        return "patchrotation"
    if "gray" in name:
        return "grayscale"
    if "bilateral" in name:
        return "bilateral"

    return name


def extract_hparams(hparams: dict) -> dict:
    out = {}

    for raw_key, new_key in HPARAM_MAP.items():
        out[new_key] = hparams.get(raw_key, np.nan)

    return out

def parse_scenario_key(key):
    if "__" not in str(key):
        return None, None

    dataset, perturb = str(key).split("__", 1)
    return dataset, normalize_perturbation(perturb)


def make_scenario_key(dataset, perturbation):
    return f"{dataset}__{normalize_perturbation(perturbation)}"


def pick(d, *keys, default=np.nan):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def build_dataset_meta(summary):
    return {
        ds["name"]: ds
        for ds in summary.get("data_config", {}).get("datasets", [])
    }


def has_validation_metrics(d):
    if not isinstance(d, dict):
        return False

    metric_keys = {
        "LV_ratio", "HFE_ratio", "ESSIM", "GC", "gc",
        "texture_score", "shape_score",
    }

    return any(k in d for k in metric_keys)


def extract_validation_metrics(obj):
    if not isinstance(obj, dict):
        return {}

    if has_validation_metrics(obj):
        return obj

    wrapper_keys = [
        "metrics",
        "scores",
        "result",
        "results",
        "validation",
        "validation_metrics",
        "perturbation_metrics",
    ]

    for key in wrapper_keys:
        if key in obj and isinstance(obj[key], dict):
            found = extract_validation_metrics(obj[key])
            if found:
                return found

    return {}


def find_validation_metrics(perturb_val, perturbation):
    """
    Find validation metrics from the actual JSON structure.

    Expected current case:
        perturbation_validation[perturbation] = {...}

    Also supports:
        perturbation_validation["results"][perturbation] = {...}
        perturbation_validation["records"][i]["perturbation"] = perturbation
        perturbation_validation[perturbation]["metrics"] = {...}
    """
    perturbation = normalize_perturbation(perturbation)

    if not perturb_val:
        return {}

    # Case 1: direct perturbation key
    if isinstance(perturb_val, dict):
        for key, value in perturb_val.items():
            if normalize_perturbation(key) == perturbation:
                found = extract_validation_metrics(value)
                if found:
                    return found

        # Case 2: common nested containers
        container_keys = [
            "results",
            "records",
            "items",
            "validations",
            "perturbations",
            "perturbation_results",
            "perturbation_validation",
            "validation_results",
        ]

        for key in container_keys:
            if key not in perturb_val:
                continue

            container = perturb_val[key]

            if isinstance(container, dict):
                for sub_key, value in container.items():
                    if normalize_perturbation(sub_key) == perturbation:
                        found = extract_validation_metrics(value)
                        if found:
                            return found

            if isinstance(container, list):
                for item in container:
                    if not isinstance(item, dict):
                        continue

                    item_perturb = pick(
                        item,
                        "perturbation",
                        "perturb",
                        "name",
                        "transform",
                        "transform_name",
                        default=None,
                    )

                    if item_perturb is not None and normalize_perturbation(item_perturb) == perturbation:
                        found = extract_validation_metrics(item)
                        if found:
                            return found

        # Case 3: recursive fallback
        for value in perturb_val.values():
            if isinstance(value, dict):
                found = find_validation_metrics(value, perturbation)
                if found:
                    return found

            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        continue

                    item_perturb = pick(
                        item,
                        "perturbation",
                        "perturb",
                        "name",
                        "transform",
                        "transform_name",
                        default=None,
                    )

                    if item_perturb is not None and normalize_perturbation(item_perturb) == perturbation:
                        found = extract_validation_metrics(item)
                        if found:
                            return found

    if isinstance(perturb_val, list):
        for item in perturb_val:
            if not isinstance(item, dict):
                continue

            item_perturb = pick(
                item,
                "perturbation",
                "perturb",
                "name",
                "transform",
                "transform_name",
                default=None,
            )

            if item_perturb is not None and normalize_perturbation(item_perturb) == perturbation:
                found = extract_validation_metrics(item)
                if found:
                    return found

    return {}


def build_scenario_df(summary, json_path):
    trial_id = json_path.parent.name
    hparams = summary.get("transform_hparams", {})
    hparam_row = extract_hparams(hparams)

    dataset_meta = build_dataset_meta(summary)
    rows = []

    for scenario in summary.get("scenarios", []):
        dataset = pick(scenario, "dataset_name", "dataset")
        perturbation = normalize_perturbation(
            pick(scenario, "perturbation", default="original")
        )
        scenario_key = make_scenario_key(dataset, perturbation)
        ds_info = dataset_meta.get(dataset, {})

        row = {
            "trial_id": trial_id,
            "trial_path": str(json_path),

            "scenario_key": scenario_key,
            "dataset": dataset,
            "perturbation": perturbation,

            "dataset_type": ds_info.get("dataset_type", np.nan),
            "domain_type": ds_info.get("domain_type", np.nan),
            "shift_type": ds_info.get("shift_type", np.nan),
            "num_classes": ds_info.get("num_classes", np.nan),

            "feature_type": PERTURB_TO_FEATURE.get(perturbation, "unknown"),
        }

        row.update(hparam_row)
        rows.append(row)

    scenario_df = pd.DataFrame(rows)

    if scenario_df.empty:
        seen = set()

        for model_result in summary.get("model_results", {}).values():
            for raw_scenario_key in model_result.get("scenario_results", {}).keys():
                dataset, perturbation = parse_scenario_key(raw_scenario_key)

                if dataset is None:
                    continue

                perturbation = normalize_perturbation(perturbation)
                scenario_key = make_scenario_key(dataset, perturbation)

                if scenario_key in seen:
                    continue

                seen.add(scenario_key)
                ds_info = dataset_meta.get(dataset, {})

                row = {
                    "trial_id": trial_id,
                    "trial_path": str(json_path),

                    "scenario_key": scenario_key,
                    "dataset": dataset,
                    "perturbation": perturbation,

                    "dataset_type": ds_info.get("dataset_type", np.nan),
                    "domain_type": ds_info.get("domain_type", np.nan),
                    "shift_type": ds_info.get("shift_type", np.nan),
                    "num_classes": ds_info.get("num_classes", np.nan),

                    "feature_type": PERTURB_TO_FEATURE.get(perturbation, "unknown"),
                }

                row.update(hparam_row)
                rows.append(row)

        scenario_df = pd.DataFrame(rows)

    return scenario_df


def build_validation_df(summary, json_path, scenario_df):
    trial_id = json_path.parent.name
    perturb_val = summary.get("perturbation_validation") or {}
    hparams = summary.get("transform_hparams", {})
    hparam_row = extract_hparams(hparams)

    rows = []

    perturbations = (
        scenario_df["perturbation"]
        .dropna()
        .map(normalize_perturbation)
        .unique()
        .tolist()
        if "perturbation" in scenario_df.columns
        else []
    )

    for perturbation in perturbations:
        metrics = find_validation_metrics(perturb_val, perturbation)

        row = {
            "trial_id": trial_id,
            "trial_path": str(json_path),

            "dataset": "imagenet",

            "perturbation": perturbation,
            "controlled_perturbation": perturbation if perturbation != "original" else "none",

            "LV_ratio": metrics.get("LV_ratio", np.nan),
            "HFE_ratio": metrics.get("HFE_ratio", np.nan),
            "ESSIM": metrics.get("ESSIM", np.nan),
            "GC": metrics.get("GC", metrics.get("gc", np.nan)),
            "texture_score": metrics.get("texture_score", np.nan),
            "shape_score": metrics.get("shape_score", np.nan),
        }

        row.update(hparam_row)
        rows.append(row)

    return pd.DataFrame(rows)


def build_result_df(summary, json_path):
    trial_id = json_path.parent.name
    rows = []

    for model_key, model_result in summary.get("model_results", {}).items():
        model_name = model_result.get("model_name", model_key)
        pretrained = model_result.get("pretrained_weight", np.nan)

        for raw_scenario_key, rec in model_result.get("scenario_results", {}).items():
            key_dataset, key_perturbation = parse_scenario_key(raw_scenario_key)

            dataset = pick(rec, "dataset_name", "dataset", default=key_dataset)
            perturbation = normalize_perturbation(
                pick(rec, "perturbation", default=key_perturbation)
            )
            scenario_key = make_scenario_key(dataset, perturbation)

            rows.append({
                "trial_id": trial_id,
                "trial_path": str(json_path),

                "model_key": model_key,
                "model": model_name,
                "pretrained": pretrained,

                "scenario_key": scenario_key,
                "raw_scenario_key": raw_scenario_key,
                "dataset": dataset,
                "perturbation": perturbation,

                "accuracy": pick(rec, "accuracy", "perturbed_accuracy"),
                "relative_accuracy": pick(
                    rec,
                    "relative_accuracy",
                    "relative_accuracy_score",
                    "rel_accuracy",
                    "rel_acc",
                ),
                "js": pick(rec, "js", "js_divergence"),
                "cka": pick(rec, "cka"),
                "drop_same": pick(
                    rec,
                    "drop_same",
                    "accuracy_drop_vs_same_dataset_clean",
                ),
                "drop_id": pick(
                    rec,
                    "drop_id",
                    "accuracy_drop_vs_id_clean",
                ),
            })

    return pd.DataFrame(rows)


def parse_one_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    scenario_df = build_scenario_df(summary, json_path)
    validation_df = build_validation_df(summary, json_path, scenario_df)
    result_df = build_result_df(summary, json_path)

    validation_for_merge = validation_df.drop(
        columns=["trial_path", "dataset", *HPARAM_COLS],
        errors="ignore",
    )

    master_df = (
        result_df
        .merge(
            scenario_df.drop(columns=["trial_path"], errors="ignore"),
            on=["trial_id", "scenario_key", "dataset", "perturbation"],
            how="left",
        )
        .merge(
            validation_for_merge,
            on=["trial_id", "perturbation"],
            how="left",
        )
    )

    return {
        "scenario_df": scenario_df,
        "validation_df": validation_df,
        "result_df": result_df,
        "master_df": master_df,
    }


def make_master_dataframe(
    root_dir=Path("Cache/MetaData"),
    save_dir=Path("analysis_outputs"),
    save_csv=True,
    strict=True,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    scenario_list = []
    validation_list = []
    result_list = []
    master_list = []

    for json_path in root_dir.rglob("experiment_summary.json"):
        try:
            parsed = parse_one_json(json_path)

            scenario_list.append(parsed["scenario_df"])
            validation_list.append(parsed["validation_df"])
            result_list.append(parsed["result_df"])
            master_list.append(parsed["master_df"])

        except Exception as e:
            print(f"[WARN] {json_path}: {e}")
            if strict:
                raise

    scenario_df = pd.concat(scenario_list, ignore_index=True) if scenario_list else pd.DataFrame()
    validation_df = pd.concat(validation_list, ignore_index=True) if validation_list else pd.DataFrame()
    result_df = pd.concat(result_list, ignore_index=True) if result_list else pd.DataFrame()
    master_df = pd.concat(master_list, ignore_index=True) if master_list else pd.DataFrame()

    # Columns to rename from merge suffixes
    rename_map = {
        "dataset_x": "dataset",
        "alpha_x": "alpha",
        "sigma_x": "sigma",
        "grid_size_x": "grid_size",
    }

    # Columns to drop
    drop_cols = [
        # duplicated by merge
        "dataset_y",
        "alpha_y",
        "sigma_y",
        "grid_size_y",

        # implementation-only metadata
        "dataset_type",
        "raw_scenario_key",

        # optional: keep only if you need debugging
        # "trial_path",
    ]


    master_df = master_df.rename(columns=rename_map).drop(columns=drop_cols, errors="ignore")

    num_cols = [
        "num_classes", "alpha", "sigma", "grid_size",
        "accuracy", "relative_accuracy", "js", "cka",
        "drop_same", "drop_id",
        "LV_ratio", "HFE_ratio", "ESSIM", "GC",
        "texture_score", "shape_score",
    ]

    for df in [scenario_df, validation_df, result_df, master_df]:
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    if "relative_accuracy" in master_df.columns:
        master_df["feature_reliance"] = 1.0 - master_df["relative_accuracy"]

    if "perturbation" in master_df.columns:
        master_df["is_localwarp"] = master_df["perturbation"].eq("localwarp")
        
    if "perturbation" in master_df.columns:
        master_df["setting_id"] = master_df.apply(make_setting_id, axis=1)

        setting_counts = (
            master_df
            .groupby("perturbation")["setting_id"]
            .nunique()
            .rename("n_settings")
            .reset_index()
        )

        master_df = master_df.merge(setting_counts, on="perturbation", how="left")
        master_df["is_sweep"] = master_df["n_settings"] > 1

    if save_csv:
        scenario_df.to_csv(save_dir / "scenario.csv", index=False)
        validation_df.to_csv(save_dir / "validation.csv", index=False)
        result_df.to_csv(save_dir / "result.csv", index=False)
        master_df.to_csv(save_dir / "master.csv", index=False)

    print("scenario_df:", scenario_df.shape)
    print("validation_df:", validation_df.shape)
    print("result_df:", result_df.shape)
    print("master_df:", master_df.shape)

    if not validation_df.empty:
        print("\nvalidation missing ratio:")
        display(validation_df[[
            "LV_ratio", "HFE_ratio", "ESSIM", "GC",
            "texture_score", "shape_score"
        ]].isna().mean())

    if not master_df.empty:
        display(master_df.head())

    return {
        "scenario_df": scenario_df,
        "validation_df": validation_df,
        "result_df": result_df,
        "master_df": master_df,
    }

# %%
result_df = make_master_dataframe(root_dir=ROOT, save_dir=SAVE_DIR)
master_df = result_df["master_df"]
validation_df = result_df["validation_df"]
scenario_df = result_df["scenario_df"]


master_df.head()

# %% [markdown]
# # 2. Sanity Check

# %%
print(master_df.columns.tolist())

# %%
print(master_df.shape)
print(master_df.columns)
print(master_df.isna().sum())

display(master_df[[
    "dataset",
    "model_key",
    "perturbation",
    "accuracy",
    "drop_same",
    "drop_id"
]].head(20))

# %%
master_df.groupby(["dataset", "model_key", "perturbation"]).size()

# %% [markdown]
# # 3. Analysis

# %% [markdown]
# ## 3.1 Perturbation Validate Check

# %%
bilateral_settings = (
    master_df[master_df["perturbation"] == "bilateral"]
    .groupby(["sigma_color", "sigma_space", "bilateral_d"])
    .size()
    .reset_index(name="count")
)

# %%
print(
    master_df[
        (master_df["perturbation"] == "bilateral") &
        (master_df["sigma_color"] == 170) &
        (master_df["sigma_space"] == 75)
    ][["model_key", "domain_type"]].value_counts()
)

# %%
master_df[master_df["is_sweep"] == True]['perturbation'].value_counts()

# %%
fixed_df = master_df[master_df["is_sweep"] == False].copy()

# 2. 논문 bilateral setting
bilateral_paper = master_df[
    (master_df["perturbation"] == "bilateral") &
    (master_df["bilateral_d"] == 11) &
    (master_df["sigma_color"] == 170) &
    (master_df["sigma_space"] == 75)
].copy()

patchshuffle_paper = master_df[
    (master_df["perturbation"] == "patchshuffle") &
    (master_df["grid_size"] == 6)
].copy()
patchrotation_paper = master_df[
    (master_df["perturbation"] == "patchrotation") &
    (master_df["grid_size"] == 6)
].copy()
localwarp_paper = master_df[
    (master_df["perturbation"] == "localwarp") &
    (master_df["alpha"] == 35) &
    (master_df["sigma"] == 3.5)
].copy()


# 3. fixed에 추가
fixed_df = pd.concat([fixed_df, bilateral_paper, patchshuffle_paper, patchrotation_paper, localwarp_paper], ignore_index=True)



fixed_summary = (
    fixed_df
    .groupby(["model_key", "dataset", "domain_type", "perturbation", "feature_type"])[
        [
            "texture_score",
            "shape_score",
            "accuracy",
            "drop_same",
            "drop_id",
            "js",
            "cka",
            "relative_accuracy",
            "feature_reliance",
        ]
    ]
    .mean()
    .reset_index()
)

display(fixed_summary)

# %% [markdown]
# ### 3.1.1 LocalWARP

# %%
sweep_df = master_df[master_df["is_sweep"] == True].copy()

sweep_summary = (
    sweep_df
    .groupby([
        "model_key",
        "dataset",
        "domain_type",
        "perturbation",
        "feature_type",
        "setting_id",
        "alpha",
        "sigma",
        "grid_size",
        "bilateral_d",
        "sigma_color",
        "sigma_space",
        "gaussian_k",
        "gaussian_sigma",
        "gray_alpha",
    ], dropna=False)[
        [
            "texture_score",
            "shape_score",
            "accuracy",
            "drop_same",
            "drop_id",
            "js",
            "cka",
            "relative_accuracy",
            "feature_reliance",
        ]
    ]
    .mean()
    .reset_index()
)

sweep_summary["relative_accuracy"] = sweep_summary["relative_accuracy"].clip(0, 1) # ensure relative accuracy is in [0, 1] 현재 1% 초과인 noisy 데이터 존재
sweep_summary["feature_reliance"] = 1.0 - sweep_summary["relative_accuracy"]
display(sweep_summary)

# %%
internal_vs_ood_sweep_summary = sweep_summary[
    (
        (sweep_summary["domain_type"] == "id") &
        (sweep_summary["dataset"] == ID_INTERNAL_DATASET)
    ) |
    (
        (sweep_summary["domain_type"] != "id") &
        (sweep_summary["dataset"] == OOD_DATASET)
    )
].copy()

lw_summary = sweep_summary[
    sweep_summary["perturbation"] == "localwarp"
].copy()

lw_internal_vs_ood_summary = internal_vs_ood_sweep_summary[
    internal_vs_ood_sweep_summary["perturbation"] == "localwarp"
].copy()

display(lw_summary)

# %%
bilateral_summary = sweep_summary[
    sweep_summary["perturbation"] == "bilateral"
].copy()

bilateral_internal_vs_ood_summary = internal_vs_ood_sweep_summary[
    internal_vs_ood_sweep_summary["perturbation"] == "bilateral"
].copy()

display(bilateral_summary)

# %%
def plot_perturbation_sweep_curves_all(
    df,
    perturbation,
    model_name="resnet50__in1k",
    domain="id",
    x_col=None,
    line_col=None,
    score_cols=("shape_score", "texture_score", "relative_accuracy"),
    model_col="model_key",
    domain_col="domain_type",
    perturb_col="perturbation",
):
    """
    Generic sweep curve plot.

    Example:
    - LocalWarp:  x_col="alpha", line_col="sigma"
    - Bilateral:  x_col="sigma_color", line_col="sigma_space"
    - Patch:      x_col="grid_size", line_col=None
    - Grayscale:  x_col="gray_alpha", line_col=None
    """

    perturbation = normalize_perturbation(perturbation)

    if x_col is None:
        default_x = {
            "localwarp": "alpha",
            "bilateral": "sigma_color",
            "patchshuffle": "grid_size",
            "patchrotation": "grid_size",
            "grayscale": "gray_alpha",
        }
        x_col = default_x.get(perturbation)

    if line_col is None:
        default_line = {
            "localwarp": "sigma",
            "bilateral": "sigma_space",
            "patchshuffle": None,
            "patchrotation": None,
            "grayscale": None,
        }
        line_col = default_line.get(perturbation)

    if x_col is None:
        raise ValueError(f"x_col must be specified for perturbation={perturbation}")

    sub = df[
        (df[perturb_col].map(normalize_perturbation) == perturbation) &
        (df[model_col] == model_name) &
        (df[domain_col] == domain)
    ].copy()

    if sub.empty:
        print(f"[WARN] No data: perturbation={perturbation}, model={model_name}, domain={domain}")
        return

    fig, axes = plt.subplots(
        1,
        len(score_cols),
        figsize=(5.2 * len(score_cols), 4),
        sharex=True,
    )

    if len(score_cols) == 1:
        axes = [axes]

    if line_col is not None and line_col in sub.columns and sub[line_col].notna().any():
        grouped = sub.groupby(line_col, dropna=False)
    else:
        grouped = [(None, sub)]

    for line_val, s_df in grouped:
        s_df = s_df.sort_values(x_col)

        label = (
            f"{line_col}={line_val}"
            if line_col is not None and pd.notna(line_val)
            else perturbation
        )

        for ax, col in zip(axes, score_cols):
            if col not in s_df.columns:
                continue

            ax.plot(
                s_df[x_col],
                s_df[col],
                marker="o",
                label=label,
            )

    for ax, col in zip(axes, score_cols):
        ax.set_title(f"{perturbation}: {x_col} vs {col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)

    axes[-1].legend(
        title=line_col if line_col is not None else "setting",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.suptitle(f"{model_name} | {domain}", y=1.03)
    plt.tight_layout()
    plt.show()
    
def make_lw_texture_high_subset(
    df,
    model_name="resnet50__in1k",
    domain="id",
    texture_quantile=0.7,
):
    sub = df[
        (df["model_key"] == model_name) &
        (df["domain_type"] == domain)
    ].copy()

    tex_th = sub["texture_score"].quantile(texture_quantile)

    tex_high = sub[sub["texture_score"] >= tex_th].copy()

    return tex_high


def plot_lw_texture_high_shape_vs_acc(
    df,
    model_name="resnet50__in1k",
    domain="id",
    texture_quantile=0.7,
):
    tex_high = make_lw_texture_high_subset(
        df,
        model_name=model_name,
        domain=domain,
        texture_quantile=texture_quantile,
    )

    plt.figure(figsize=(5, 4))

    sns.scatterplot(
        data=tex_high,
        x="shape_score",
        y="relative_accuracy",
        hue="sigma",
        style="alpha",
        alpha=0.85,
    )

    sns.regplot(
        data=tex_high,
        x="shape_score",
        y="relative_accuracy",
        scatter=False,
    )

    plt.title(f"LW texture-high subset | {model_name} | {domain}")
    plt.xlabel("shape_score")
    plt.ylabel("relative_accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%
plot_perturbation_sweep_curves_all(
    internal_vs_ood_sweep_summary,
    perturbation="localwarp",
    model_name="resnet50__in1k",
    domain="id",
)
plot_perturbation_sweep_curves_all(
    internal_vs_ood_sweep_summary,
    perturbation="localwarp",
    model_name="resnet50__in1k",
    domain="natural_ood",
)
plot_perturbation_sweep_curves_all(
    internal_vs_ood_sweep_summary,
    perturbation="localwarp",
    model_name="vit-b__augreg_in1k",
    domain="id",
)
plot_perturbation_sweep_curves_all(
    internal_vs_ood_sweep_summary,
    perturbation="localwarp",
    model_name="vit-b__augreg_in1k",
    domain="natural_ood",
)    


# %%
plot_lw_texture_high_shape_vs_acc(
    lw_internal_vs_ood_summary,
    model_name="resnet50__in1k",
    domain="id",
    texture_quantile=0.5,
)

plot_lw_texture_high_shape_vs_acc(
    lw_internal_vs_ood_summary,
    model_name="vit-b__augreg_in1k",
    domain="id",
    texture_quantile=0.5,
)

# %%
plot_perturbation_sweep_curves_all(
    internal_vs_ood_sweep_summary,
    perturbation="bilateral",
    model_name="resnet50__in1k",
    domain="id",
)
plot_perturbation_sweep_curves_all(
    internal_vs_ood_sweep_summary,
    perturbation="bilateral",
    model_name="resnet50__in1k",
    domain="natural_ood",
)
plot_perturbation_sweep_curves_all(
    internal_vs_ood_sweep_summary,
    perturbation="bilateral",
    model_name="vit-b__augreg_in1k",
    domain="id",
)
plot_perturbation_sweep_curves_all(
    internal_vs_ood_sweep_summary,
    perturbation="bilateral",
    model_name="vit-b__augreg_in1k",
    domain="natural_ood",
)    


# %%
def plot_score_collapse(
    df,
    score_col="shape_score",
    perf_col="relative_accuracy",
    model_col="model_key",
    domain_col="domain_type",
):
    models = sorted(df[model_col].dropna().unique())
    domains = sorted(df[domain_col].dropna().unique())

    fig, axes = plt.subplots(
        len(models),
        len(domains),
        figsize=(6 * len(domains), 4 * len(models)),
        sharex=True,
        sharey=True,
    )

    if len(models) == 1:
        axes = np.array([axes])
    if len(domains) == 1:
        axes = axes.reshape(len(models), 1)

    for i, model in enumerate(models):
        for j, domain in enumerate(domains):
            ax = axes[i, j]

            sub = df[
                (df[model_col] == model) &
                (df[domain_col] == domain)
            ]

            sns.scatterplot(
                data=sub,
                x=score_col,
                y=perf_col,
                hue="perturbation",
                style="perturbation",
                ax=ax,
                alpha=0.85,
            )

            if len(sub.dropna(subset=[score_col, perf_col])) >= 3:
                sns.regplot(
                    data=sub,
                    x=score_col,
                    y=perf_col,
                    scatter=False,
                    ax=ax,
                )

            ax.set_title(f"{model} | {domain}")
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    
def pivot_metric(df, value_col, alpha_col="alpha", sigma_col="sigma"):
    return (
        df.pivot_table(
            index=sigma_col,
            columns=alpha_col,
            values=value_col,
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )


def _imshow_real_axis(
    ax,
    pivot,
    title,
    xlabel="alpha",
    ylabel="sigma",
    vmin=None,
    vmax=None,
):
    alphas = pivot.columns.to_numpy(dtype=float)
    sigmas = pivot.index.to_numpy(dtype=float)

    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            alphas.min(),
            alphas.max(),
            sigmas.min(),
            sigmas.max(),
        ],
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(alphas)
    ax.set_yticks(sigmas)

    return im


def plot_lw_heatmap(
    df,
    perf_col="relative_accuracy",
    score_col="shape_score",
    model_col="model_key",
    domain_col="domain_type",
    alpha_col="alpha",
    sigma_col="sigma",
    domains=("id", "natural_ood"),
    models=None,
    score_vmin=0.0,
    
    score_vmax=1.0,
    perf_vmin=0.0,
    perf_vmax=1.0,
):
    if models is None:
        models = sorted(df[model_col].dropna().unique())

    assert len(domains) == 2, "2 domains required"
    assert len(models) == 2, "2 models required"

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for r, domain in enumerate(domains):
        ddf = df[df[domain_col] == domain]

        # (1) shape score
        shape_df = ddf.drop_duplicates(subset=[alpha_col, sigma_col])
        shape_pivot = pivot_metric(shape_df, score_col, alpha_col, sigma_col)

        im = _imshow_real_axis(
            axes[r, 0],
            shape_pivot,
            title=f"{domain.upper()} | {score_col}",
            vmin=score_vmin,
            vmax=score_vmax,
        )
        fig.colorbar(im, ax=axes[r, 0])

        # (2, 3) model performance
        for c, model in enumerate(models, start=1):
            mdf = ddf[ddf[model_col] == model]
            perf_pivot = pivot_metric(mdf, perf_col, alpha_col, sigma_col)

            im = _imshow_real_axis(
                axes[r, c],
                perf_pivot,
                title=f"{domain.upper()} | {model} {perf_col}",
                vmin=perf_vmin,
                vmax=perf_vmax,
            )
            fig.colorbar(im, ax=axes[r, c])

    plt.tight_layout()
    plt.show()
      
def plot_lw_iso(
    df,
    model_name,
    perf_col="rel_acc",
    score_col="shape_score",
    model_col="model_key",
    domain_col="domain_type",
    domain="id",
    alpha_col="alpha",
    sigma_col="sigma",
):
    ddf = df[df[domain_col] == domain]
    mdf = ddf[ddf[model_col] == model_name]

    shape_df = ddf.drop_duplicates(subset=[alpha_col, sigma_col])

    shape_pivot = pivot_metric(shape_df, score_col, alpha_col, sigma_col)
    perf_pivot = pivot_metric(mdf, perf_col, alpha_col, sigma_col)

    alphas = shape_pivot.columns.to_numpy(dtype=float)
    sigmas = shape_pivot.index.to_numpy(dtype=float)
    A, S = np.meshgrid(alphas, sigmas)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        perf_pivot.values,
        origin="lower",
        aspect="auto",
        extent=[alphas.min(), alphas.max(), sigmas.min(), sigmas.max()],
    )

    cs = ax.contour(A, S, shape_pivot.values, levels=5)
    ax.clabel(cs, inline=True, fontsize=8)

    ax.set_title(f"{domain.upper()} | {model_name} ({perf_col})")
    ax.set_xlabel("alpha")
    ax.set_ylabel("sigma")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


# %%
shape_collapse_df = pd.concat([
    lw_internal_vs_ood_summary,
    fixed_summary[
        fixed_summary["perturbation"].isin(["patchshuffle", "patchrotation"]) &
        (
            (
                (fixed_summary["domain_type"] == "id") &
                (fixed_summary["dataset"] == ID_INTERNAL_DATASET)
            ) |
            (
                (fixed_summary["domain_type"] != "id") &
                (fixed_summary["dataset"] == OOD_DATASET)
            )
        )
    ],
], ignore_index=True)

texture_control_df = pd.concat([
    bilateral_summary[
        (
            (bilateral_summary["domain_type"] == "id") &
            (bilateral_summary["dataset"] == ID_INTERNAL_DATASET)
        ) |
        (
            (bilateral_summary["domain_type"] != "id") &
            (bilateral_summary["dataset"] == OOD_DATASET)
        )
    ],
    # 나중에 gaussian 있으면 추가
    # sweep_summary[sweep_summary["perturbation"] == "gaussian"]
], ignore_index=True)

print("Shape collapse candidates:")


# %% [markdown]
# ### 3.1.2 1D collapse curve (Shape_score vs drop_Acc)

# %%
# rel_acc 기반 (논문용)
plot_score_collapse(
    shape_collapse_df,
    score_col="shape_score",
    perf_col="relative_accuracy",
)
plot_score_collapse(
    shape_collapse_df,
    score_col="shape_score",
    perf_col="drop_same",
)


# %% [markdown]
# **Slope**

# %% [markdown]
# ### 3.1.2-b 1D collapse curve (texture_score vs drop_Acc)

# %%
# rel_acc 기반 (논문용)
plot_score_collapse(shape_collapse_df, score_col="texture_score",perf_col="relative_accuracy")
plot_score_collapse(shape_collapse_df, score_col="texture_score", perf_col="drop_same")

# %%
plot_score_collapse(
    texture_control_df,
    score_col="shape_score",
    perf_col="relative_accuracy",
)
plot_score_collapse(
    texture_control_df,
    score_col="shape_score",
    perf_col="drop_same",
)

# %%
plot_score_collapse(
    texture_control_df,
    score_col="texture_score",
    perf_col="relative_accuracy",
)
plot_score_collapse(
    texture_control_df,
    score_col="texture_score",
    perf_col="drop_same",
)

# %%
from scipy.stats import pearsonr, spearmanr

def plot_feature_sensitivity_figure(df:pd.DataFrame, model_col="model_key", x_col="shape_score", y_col="relative_accuracy"):
    models = sorted(df[model_col].dropna().unique())
    features = ["shape_score", "texture_score"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)

    for i, model in enumerate(models):
        sub = df[df["model_key"] == model]

        for j, feat in enumerate(features):
            ax = axes[i, j]

            x = sub[feat].values
            y = sub["relative_accuracy"].values

            ax.scatter(x, y, alpha=0.7)

            # regression line
            if len(x) > 2:
                m, b = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = m * x_line + b
                ax.plot(x_line, y_line)

                ax.text(
                    0.05, 0.05,
                    f"slope={m:.2f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7)
                )

            ax.set_title(f"{model} | {feat}")
            ax.set_xlabel(feat)
            ax.set_ylabel("relative_accuracy")
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    
def compute_sensitivity_table(df, x_lable="feature_score", y_label="relative_accuracy", model_col="model_key"):
    rows = []

    for model in df["model_key"].unique():
        sub = df[df["model_key"] == model]

        for feat in ["shape_score", "texture_score"]:
            data = sub[[feat, "relative_accuracy"]].dropna()

            if len(data) < 3:
                continue

            x = data[feat].values
            y = data["relative_accuracy"].values

            m, _ = np.polyfit(x, y, 1)
            pr, _ = pearsonr(x, y)
            sr, _ = spearmanr(x, y)

            rows.append({
                "model": model,
                "feature": feat,
                "slope": m,
                "abs_slope": abs(m),
                "pearson_r": pr,
                "spearman_r": sr,
                "r2": pr**2,
            })

    return pd.DataFrame(rows)

# %%
plot_feature_sensitivity_figure(shape_collapse_df)
compute_sensitivity_table(shape_collapse_df)

# %%
plot_feature_sensitivity_figure(texture_control_df)
compute_sensitivity_table(texture_control_df)

# %%
shape_sensitivity = compute_sensitivity_table(shape_collapse_df)
texture_sensitivity = compute_sensitivity_table(texture_control_df)

print("Shape Sensitivity:")
display(shape_sensitivity)
print()
print("Texture Sensitivity:")
display(texture_sensitivity)

# %% [markdown]
# ### 3.1.3 HeatMap
# - 2D heatmap (X-axis: alpha, Y-axis: sigma Color: Acc or drop_same)
# - 2D heatmap (X-axis: alpha, Y-axis: sigma Color: shape score) 

# %%
plot_lw_heatmap(lw_internal_vs_ood_summary, perf_col="relative_accuracy")

# %% [markdown]
# ### 3.1.4 iso-contour
# - same shape_score line 따라가면 accuracy도 유사한가?

# %%
plot_lw_iso(lw_internal_vs_ood_summary, model_name="resnet50__in1k", perf_col="relative_accuracy", domain="id")
plot_lw_iso(lw_internal_vs_ood_summary, model_name="resnet50__in1k", perf_col="relative_accuracy", domain="natural_ood")

plot_lw_iso(lw_internal_vs_ood_summary, model_name="vit-b__augreg_in1k", perf_col="relative_accuracy", domain="id")
plot_lw_iso(lw_internal_vs_ood_summary, model_name="vit-b__augreg_in1k", perf_col="relative_accuracy", domain="natural_ood")

# %% [markdown]
# ## 3.2 Feature reliance vs OOD accuracy

# %%
num_classes = 200
chance = 1.0 / num_classes

id_clean_acc = (
    fixed_summary[
        (fixed_summary["dataset"] == ID_OOD_BASELINE_DATASET) &
        (fixed_summary["perturbation"] == "original")
    ]
    .set_index("model_key")["accuracy"]
)

ood_clean_acc = (
    fixed_summary[
        (fixed_summary["dataset"] == OOD_DATASET) &
        (fixed_summary["perturbation"] == "original")
    ]
    .set_index("model_key")["accuracy"]
)

ood_rel_acc = (ood_clean_acc - chance) / (id_clean_acc - chance)
id_ood_gap = 1.0 - ood_rel_acc

# shape / texture 분리
shape_df = fixed_summary[
    (fixed_summary["feature_type"] == "shape") &
    (fixed_summary["dataset"] == ID_OOD_BASELINE_DATASET)
]

texture_df = fixed_summary[
    (fixed_summary["feature_type"] == "texture") &
    (fixed_summary["dataset"] == ID_OOD_BASELINE_DATASET)
]

# 모델별 평균 reliance
shape_rel = shape_df.groupby("model_key")["feature_reliance"].mean()
texture_rel = texture_df.groupby("model_key")["feature_reliance"].mean()

# %%
ood_df = fixed_summary[
    (fixed_summary["perturbation"] == "original") &
    (fixed_summary["dataset"] == OOD_DATASET)
]

ood_perf = ood_df.set_index("model_key")["accuracy"]

# %%
analysis_df = pd.concat(
    [
        shape_rel.rename("shape_rel_id"),
        ood_rel_acc.rename("ood_clean_rel_acc"),
        id_ood_gap.rename("rel_id_ood_gap"),
    ],
    axis=1,
)

display(analysis_df)

# %%
plt.figure(figsize=(5,4))

for model, row in analysis_df.iterrows():
    plt.scatter(row["shape_rel_id"], row["rel_id_ood_gap"])
    plt.text(row["shape_rel_id"], row["rel_id_ood_gap"], model)

plt.xlabel("Shape reliance (ID)")
plt.ylabel("rel_id_ood_gap")
plt.title("Feature reliance vs OOD performance")

plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## 3.3 feature vulnerability profile
# 
# 1. 각 모델이 어떤 feature suppression에 가장 취약한가?
# 2. 그 취약성이 ID/OOD에서 일관적인가?
# 
# 
# 인지 확인

# %%
def get_target_score(row):
    if row["feature_type"] == "shape":
        return row["shape_score"]
    elif row["feature_type"] == "texture":
        return row["texture_score"]
    else:
        return np.nan  # color 등은 제외

profile_df = fixed_summary[
    (
        (fixed_summary["domain_type"] == "id") &
        (fixed_summary["dataset"] == ID_OOD_BASELINE_DATASET)
    ) |
    (
        (fixed_summary["domain_type"] != "id") &
        (fixed_summary["dataset"] == OOD_DATASET)
    )
].copy()

# target feature score
profile_df["target_score"] = profile_df.apply(get_target_score, axis=1)

# suppression strength (얼마나 많이 깨졌는지)
profile_df["suppression_strength"] = 1.0 - profile_df["target_score"]

# 안정성 필터 (너무 약한 perturbation 제거)
profile_df = profile_df[profile_df["suppression_strength"] > 1e-3]

# 핵심: normalized vulnerability 1의 texture or shape score가 떨어질때 상대적으로 얼마나 accuracy가 떨어지는지
profile_df["norm_vulnerability"] = (
    profile_df["feature_reliance"] / profile_df["suppression_strength"]
)

# 집계
feature_stats = (
    profile_df
    .groupby(["model_key", "domain_type", "feature_type"])["norm_vulnerability"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

display(feature_stats)

# %%

def plot_feature_vulnerability_with_error(feature_stats):
    models = sorted(feature_stats["model_key"].unique())
    domains = sorted(feature_stats["domain_type"].unique())
    features = ["shape", "texture", "color"]

    fig, axes = plt.subplots(
        len(domains),
        len(models),
        figsize=(5 * len(models), 4 * len(domains)),
        sharey=True,
    )

    if len(domains) == 1:
        axes = np.array([axes])
    if len(models) == 1:
        axes = axes.reshape(len(domains), 1)

    for r, domain in enumerate(domains):
        for c, model in enumerate(models):
            ax = axes[r, c]

            sub = feature_stats[
                (feature_stats["model_key"] == model) &
                (feature_stats["domain_type"] == domain)
            ]

            means = []
            stds = []

            for f in features:
                row = sub[sub["feature_type"] == f]
                if len(row) > 0:
                    means.append(row["mean"].values[0])
                    stds.append(row["std"].values[0])
                else:
                    means.append(0)
                    stds.append(0)

            x = np.arange(len(features))

            ax.bar(
                x,
                means,
                yerr=stds,
                capsize=5,
            )

            ax.set_xticks(x)
            ax.set_xticklabels(features)
            ax.set_title(f"{model} | {domain}")
            ax.set_ylabel("Normalized Vulnerability")
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
    
plot_feature_vulnerability_with_error(feature_stats)


