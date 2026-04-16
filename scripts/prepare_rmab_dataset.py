#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


VITAL_ITEMS = {220045: "heart_rate", 220181: "map", 220210: "resp_rate", 220277: "spo2"}
GCS_ITEMS = [220739, 223900, 223901]
LAB_ITEMS = {50813: "lactate", 50912: "creatinine"}


@dataclass
class BuildConfig:
    state_spec: str = "4var"
    alpha: float = 2.0
    min_support: int = 10
    bootstraps: int = 200
    seed: int = 42


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def freeze_baseline(repo_root: Path, out_dir: Path) -> Dict[str, str]:
    baseline_dir = out_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, str] = {}
    for fname in ["P_icu.npy", "P_nonicu.npy"]:
        src = repo_root / fname
        if not src.exists():
            continue
        dst = baseline_dir / f"{src.stem}_baseline.npy"
        shutil.copy2(src, dst)
        manifest[str(dst.relative_to(out_dir))] = _sha256(dst)
    with (baseline_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def ensure_mimic_root(repo_root: Path, demo_zip: Path, mimic_root: Path | None) -> Path:
    if mimic_root:
        if (mimic_root / "hosp").exists() and (mimic_root / "icu").exists():
            return mimic_root
        raise FileNotFoundError(f"Invalid mimic root: {mimic_root}")

    extracted = repo_root / "data" / "mimic-iv-demo"
    if (extracted / "mimic-iv-clinical-database-demo-2.2" / "hosp").exists():
        return extracted / "mimic-iv-clinical-database-demo-2.2"

    if not demo_zip.exists():
        raise FileNotFoundError("No --mimic-root provided and demo zip not found")

    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(demo_zip, "r") as zf:
        zf.extractall(extracted)
    root = extracted / "mimic-iv-clinical-database-demo-2.2"
    if not (root / "hosp").exists():
        raise FileNotFoundError("Could not locate extracted MIMIC-IV folders")
    return root


def expand_to_daily(df: pd.DataFrame, start_col: str, end_col: str, keys: list[str]) -> pd.DataFrame:
    out = df[keys + [start_col, end_col]].copy()
    out["date"] = out.apply(lambda r: pd.date_range(r[start_col].normalize(), r[end_col].normalize(), freq="D"), axis=1)
    out = out.explode("date", ignore_index=True)
    return out[keys + ["date"]]


def build_features(mimic_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hosp = mimic_root / "hosp"
    icu = mimic_root / "icu"

    patients = pd.read_csv(hosp / "patients.csv.gz", compression="gzip", usecols=["subject_id", "gender", "anchor_age"])
    admissions = pd.read_csv(
        hosp / "admissions.csv.gz",
        compression="gzip",
        parse_dates=["admittime", "dischtime"],
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "hospital_expire_flag"],
        low_memory=False,
    )
    icustays = pd.read_csv(
        icu / "icustays.csv.gz",
        compression="gzip",
        parse_dates=["intime", "outtime"],
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    )
    chartevents = pd.read_csv(
        icu / "chartevents.csv.gz",
        compression="gzip",
        usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
        low_memory=False,
    )
    labevents = pd.read_csv(
        hosp / "labevents.csv.gz",
        compression="gzip",
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
        low_memory=False,
    )

    adm = admissions.dropna(subset=["admittime", "dischtime"]).copy()
    daily = expand_to_daily(adm, "admittime", "dischtime", ["subject_id", "hadm_id", "hospital_expire_flag"])

    icu_days = expand_to_daily(icustays.dropna(subset=["intime", "outtime"]), "intime", "outtime", ["subject_id", "hadm_id"])
    icu_days["is_icu"] = 1
    icu_days = icu_days.drop_duplicates(["subject_id", "hadm_id", "date"])

    daily = daily.merge(icu_days, on=["subject_id", "hadm_id", "date"], how="left")
    daily["is_icu"] = daily["is_icu"].fillna(0).astype(int)

    ce = chartevents[chartevents["itemid"].isin(list(VITAL_ITEMS) + GCS_ITEMS)].copy()
    ce["date"] = ce["charttime"].dt.normalize()

    ce_non_gcs = ce[ce["itemid"].isin(VITAL_ITEMS)].copy()
    ce_non_gcs["vital"] = ce_non_gcs["itemid"].map(VITAL_ITEMS)
    vitals_daily = (
        ce_non_gcs.groupby(["subject_id", "date", "vital"])["valuenum"]
        .mean()
        .reset_index()
        .pivot_table(index=["subject_id", "date"], columns="vital", values="valuenum", aggfunc="mean")
        .reset_index()
    )
    vitals_daily.columns.name = None

    ce_gcs = ce[ce["itemid"].isin(GCS_ITEMS)]
    gcs_moment = ce_gcs.groupby(["subject_id", "stay_id", "charttime"])["valuenum"].sum().reset_index()
    gcs_moment = gcs_moment.rename(columns={"valuenum": "gcs_total"})
    gcs_moment = gcs_moment[(gcs_moment["gcs_total"] >= 3) & (gcs_moment["gcs_total"] <= 15)]
    gcs_moment["date"] = gcs_moment["charttime"].dt.normalize()
    gcs_daily = gcs_moment.groupby(["subject_id", "date"])["gcs_total"].mean().reset_index()
    vitals_daily = vitals_daily.merge(gcs_daily, on=["subject_id", "date"], how="outer")

    le = labevents[labevents["itemid"].isin(LAB_ITEMS)].copy()
    le["lab"] = le["itemid"].map(LAB_ITEMS)
    le["date"] = le["charttime"].dt.normalize()
    labs_daily_raw = (
        le.groupby(["subject_id", "date", "lab"])["valuenum"]
        .mean()
        .reset_index()
        .pivot_table(index=["subject_id", "date"], columns="lab", values="valuenum", aggfunc="mean")
        .reset_index()
    )
    labs_daily_raw.columns.name = None

    patient_dates = daily[["subject_id", "date"]].drop_duplicates()
    labs_daily = patient_dates.merge(labs_daily_raw, on=["subject_id", "date"], how="left")
    labs_daily = labs_daily.sort_values(["subject_id", "date"])
    labs_daily[["lactate", "creatinine"]] = labs_daily.groupby("subject_id")[["lactate", "creatinine"]].ffill()

    features = daily[["subject_id", "hadm_id", "date", "is_icu", "hospital_expire_flag"]].copy()
    features = features.merge(vitals_daily, on=["subject_id", "date"], how="left")
    features = features.merge(labs_daily, on=["subject_id", "date"], how="left")
    features = features.merge(patients, on="subject_id", how="left")

    feature_cols = ["map", "spo2", "gcs_total", "lactate", "heart_rate", "resp_rate", "creatinine", "anchor_age"]
    for col in feature_cols:
        if col not in features:
            features[col] = np.nan
    features = features.sort_values(["subject_id", "date"]).reset_index(drop=True)
    features[feature_cols] = features.groupby("subject_id")[feature_cols].ffill()
    medians = features[feature_cols].median(numeric_only=True)
    features[feature_cols] = features[feature_cols].fillna(medians)

    return features, admissions


def discretize_state(features: pd.DataFrame, state_spec: str) -> Tuple[pd.DataFrame, int, int]:
    df = features.copy()
    df["map_bin"] = pd.cut(df["map"], bins=[-np.inf, 65, 100, np.inf], labels=[0, 1, 2]).astype(int)
    df["spo2_bin"] = pd.cut(df["spo2"], bins=[-np.inf, 90, 95, np.inf], labels=[0, 1, 2]).astype(int)
    df["gcs_bin"] = pd.cut(df["gcs_total"], bins=[2, 8, 12, 15], labels=[0, 1, 2]).astype(int)
    df["lactate_bin"] = pd.cut(df["lactate"], bins=[-np.inf, 2, 4, np.inf], labels=[2, 1, 0]).astype(int)

    if state_spec == "4var":
        vars_ = ["map_bin", "spo2_bin", "gcs_bin", "lactate_bin"]
    else:
        df["hr_bin"] = pd.cut(df["heart_rate"], bins=[-np.inf, 60, 110, np.inf], labels=[1, 2, 0]).astype(int)
        df["rr_bin"] = pd.cut(df["resp_rate"], bins=[-np.inf, 12, 24, np.inf], labels=[1, 2, 0]).astype(int)
        df["creat_bin"] = pd.cut(df["creatinine"], bins=[-np.inf, 1.2, 2.0, np.inf], labels=[2, 1, 0]).astype(int)
        df["age_bin"] = pd.cut(df["anchor_age"], bins=[-np.inf, 50, 70, np.inf], labels=[2, 1, 0]).astype(int)
        vars_ = ["map_bin", "spo2_bin", "gcs_bin", "lactate_bin", "hr_bin", "rr_bin", "creat_bin", "age_bin"]

    n_live_states = 3 ** len(vars_)
    multipliers = [3 ** p for p in range(len(vars_) - 1, -1, -1)]
    state = np.zeros(len(df), dtype=int)
    for c, m in zip(vars_, multipliers):
        state += df[c].to_numpy(dtype=int) * m
    death_state = n_live_states
    df["state"] = state

    last_day = df.groupby("hadm_id")["date"].transform("max")
    died_mask = (df["hospital_expire_flag"] == 1) & (df["date"] == last_day)
    df.loc[died_mask, "state"] = death_state
    return df, n_live_states + 1, death_state


def build_transitions(states_df: pd.DataFrame) -> pd.DataFrame:
    s = states_df.sort_values(["subject_id", "hadm_id", "date"]).reset_index(drop=True).copy()
    s["state_next"] = s.groupby(["subject_id", "hadm_id"])["state"].shift(-1)
    t = s.dropna(subset=["state_next"]).copy()
    t["state"] = t["state"].astype(int)
    t["state_next"] = t["state_next"].astype(int)
    return t


def build_counts(transitions: pd.DataFrame, n_states: int) -> Tuple[np.ndarray, np.ndarray]:
    c_icu = np.zeros((n_states, n_states), dtype=float)
    c_non = np.zeros((n_states, n_states), dtype=float)
    g_icu = transitions[transitions["is_icu"] == 1].groupby(["state", "state_next"]).size()
    g_non = transitions[transitions["is_icu"] == 0].groupby(["state", "state_next"]).size()
    for (s, sn), v in g_icu.items():
        c_icu[int(s), int(sn)] = float(v)
    for (s, sn), v in g_non.items():
        c_non[int(s), int(sn)] = float(v)
    return c_icu, c_non


def smooth_counts(counts: np.ndarray, alpha: float, min_support: int, death_state: int) -> np.ndarray:
    probs = np.zeros_like(counts, dtype=float)
    global_prior = counts.sum(axis=0)
    global_prior = global_prior / global_prior.sum() if global_prior.sum() > 0 else np.ones(counts.shape[1]) / counts.shape[1]
    row_sums = counts.sum(axis=1)
    for i in range(counts.shape[0]):
        if i == death_state:
            probs[i, death_state] = 1.0
            continue
        if row_sums[i] >= min_support:
            probs[i] = (counts[i] + alpha * global_prior) / (row_sums[i] + alpha)
        else:
            probs[i] = global_prior
    return probs


def bootstrap_self_loop_ci(transitions: pd.DataFrame, n_states: int, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for action in [0, 1]:
        t = transitions[transitions["is_icu"] == action]
        for s in range(n_states):
            ts = t[t["state"] == s]
            n = len(ts)
            if n == 0:
                rows.append({"action": action, "state": s, "support": 0, "self_loop_low": np.nan, "self_loop_high": np.nan})
                continue
            idx = ts["state_next"].to_numpy()
            vals = np.empty(n_boot, dtype=float)
            for b in range(n_boot):
                sample = rng.choice(idx, size=n, replace=True)
                vals[b] = float((sample == s).mean())
            rows.append(
                {
                    "action": action,
                    "state": s,
                    "support": n,
                    "self_loop_low": float(np.quantile(vals, 0.025)),
                    "self_loop_high": float(np.quantile(vals, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def build_initial_cohort(states_df: pd.DataFrame, death_state: int) -> pd.DataFrame:
    first = states_df.sort_values(["hadm_id", "date"]).groupby("hadm_id", as_index=False).first()
    first = first[["hadm_id", "subject_id", "state", "gender", "anchor_age"]].copy()
    first["is_female"] = (first["gender"] == "F").astype(int)
    first["age_group"] = pd.cut(first["anchor_age"], bins=[-np.inf, 50, 70, np.inf], labels=["<50", "50-70", ">70"])
    first = first[first["state"] != death_state]
    return first


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MIMIC data artifacts for RMAB")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--demo-zip", default="data/mimic-iv-demo.zip")
    parser.add_argument("--mimic-root", default=None)
    parser.add_argument("--out-dir", default="artifacts/rmab")
    parser.add_argument("--state-spec", choices=["4var", "8var"], default="4var")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--bootstraps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = BuildConfig(args.state_spec, args.alpha, args.min_support, args.bootstraps, args.seed)
    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_manifest = freeze_baseline(repo_root, out_dir)
    mimic_root = ensure_mimic_root(repo_root, (repo_root / args.demo_zip).resolve(), Path(args.mimic_root).resolve() if args.mimic_root else None)

    features, _ = build_features(mimic_root)
    states_df, n_states, death_state = discretize_state(features, cfg.state_spec)
    transitions = build_transitions(states_df)
    c_icu, c_non = build_counts(transitions, n_states)

    P_icu = smooth_counts(c_icu, cfg.alpha, cfg.min_support, death_state)
    P_non = smooth_counts(c_non, cfg.alpha, cfg.min_support, death_state)

    matrices_dir = out_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    np.save(matrices_dir / "P_icu_smoothed.npy", P_icu)
    np.save(matrices_dir / "P_nonicu_smoothed.npy", P_non)
    np.save(matrices_dir / "count_icu.npy", c_icu)
    np.save(matrices_dir / "count_nonicu.npy", c_non)

    data_dir = out_dir / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    states_df.to_csv(data_dir / "states_daily.csv.gz", index=False)
    transitions.to_csv(data_dir / "transitions.csv.gz", index=False)
    build_initial_cohort(states_df, death_state).to_csv(data_dir / "initial_cohort.csv", index=False)

    row_support = pd.DataFrame(
        {
            "state": np.arange(n_states),
            "support_icu": c_icu.sum(axis=1).astype(int),
            "support_nonicu": c_non.sum(axis=1).astype(int),
        }
    )
    row_support["support_total"] = row_support["support_icu"] + row_support["support_nonicu"]
    row_support["low_support"] = (row_support["support_total"] < cfg.min_support).astype(int)
    row_support.to_csv(out_dir / "row_support.csv", index=False)

    ci = bootstrap_self_loop_ci(transitions, n_states, cfg.bootstraps, cfg.seed)
    ci.to_csv(out_dir / "bootstrap_self_loop_ci.csv", index=False)

    diagnostics = {
        "config": asdict(cfg),
        "n_states": int(n_states),
        "death_state": int(death_state),
        "states_observed": int(((c_icu.sum(axis=1) + c_non.sum(axis=1)) > 0).sum()),
        "total_transitions": int(len(transitions)),
        "icu_transitions": int((transitions["is_icu"] == 1).sum()),
        "nonicu_transitions": int((transitions["is_icu"] == 0).sum()),
        "low_support_rows": int(row_support["low_support"].sum()),
        "baseline_manifest": baseline_manifest,
    }
    with (out_dir / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print(json.dumps(diagnostics, indent=2))
    print(f"Saved artifacts under: {out_dir}")


if __name__ == "__main__":
    main()
