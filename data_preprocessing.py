from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parent


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, I_future: np.ndarray, soc_start: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.I_future = torch.from_numpy(I_future)
        self.soc_start = torch.from_numpy(soc_start)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.I_future[idx], self.soc_start[idx]


def load_config(config_path: str | Path = ROOT_DIR / "config.yaml") -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT_DIR / path


def load_dataframe(cfg: dict[str, Any], generalization_test_only: bool = False) -> pd.DataFrame:
    data_path = resolve_path(cfg["paths"]["external_test_file"] if generalization_test_only else cfg["paths"]["data_file"])
    return pd.read_csv(data_path, encoding=cfg["data"]["encoding"])


def split_segments(df: pd.DataFrame, time_column: str, max_gap_seconds: int, min_len: int) -> list[pd.DataFrame]:
    data = df.copy()
    data[time_column] = pd.to_datetime(data[time_column], format="%Y-%m-%d %H:%M:%S")
    seg_flags = data[time_column].diff().dt.total_seconds().fillna(0).gt(max_gap_seconds)
    data["seg_id"] = seg_flags.cumsum()
    return [g.reset_index(drop=True) for _, g in data.groupby("seg_id") if len(g) >= min_len]


def fit_scalers(segments: list[pd.DataFrame], feature_columns: list[str], target_columns: list[str]):
    all_feat_df = pd.concat([seg[feature_columns] for seg in segments], ignore_index=True)
    all_tgt_df = pd.concat([seg[target_columns] for seg in segments], ignore_index=True)
    feature_scaler = MinMaxScaler().fit(all_feat_df)
    target_scaler = MinMaxScaler().fit(all_tgt_df)
    return feature_scaler, target_scaler


def generate_samples(
    segment_list: list[pd.DataFrame],
    in_steps: int,
    out_steps: int,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    feature_columns: list[str],
    target_columns: list[str],
    current_column: str,
    soc_column: str,
    step: int = 1,
):
    X, y, I_future, soc_start = [], [], [], []
    for seg in segment_list:
        f_scaled = feature_scaler.transform(seg[feature_columns])
        t_scaled = target_scaler.transform(seg[target_columns])
        currents_unscaled = seg[current_column].values
        soc_unscaled = seg[soc_column].values
        for i in range(0, len(seg) - in_steps - out_steps + 1, step):
            X.append(f_scaled[i : i + in_steps])
            y.append(t_scaled[i + in_steps : i + in_steps + out_steps])
            I_future.append(currents_unscaled[i + in_steps : i + in_steps + out_steps])
            soc_start.append(soc_unscaled[i + in_steps - 1])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(I_future, dtype=np.float32),
        np.asarray(soc_start, dtype=np.float32),
    )


def build_timestamp_index(
    test_segments: list[pd.DataFrame],
    time_column: str,
    min_len: int,
    input_steps: int,
    output_steps: int,
) -> list[pd.Timestamp]:
    timestamps: list[pd.Timestamp] = []
    for seg in test_segments:
        t_arr = seg[time_column].values
        for i in range(0, len(seg) - min_len + 1, output_steps):
            timestamps.extend(t_arr[i + input_steps : i + min_len])
    return timestamps


def prepare_data(cfg: dict[str, Any], generalization_test_only: bool = False):
    if generalization_test_only:
        raise NotImplementedError("Generalization-only mode is not implemented.")

    data_cfg = cfg["data"]
    seq_cfg = cfg["sequence"]
    ds_cfg = cfg["dataset"]

    df = load_dataframe(cfg, generalization_test_only=False)
    min_len = seq_cfg["input_steps"] + seq_cfg["output_steps"]
    segments = split_segments(df, data_cfg["time_column"], data_cfg["max_gap_seconds"], min_len)

    if not segments:
        raise ValueError("No valid segments were found in the input data.")

    feature_scaler, target_scaler = fit_scalers(segments, ds_cfg["feature_columns"], ds_cfg["target_columns"])

    n_total = len(segments)
    train_end = int(data_cfg["train_ratio"] * n_total)
    val_end = train_end + int(data_cfg["val_ratio"] * n_total)
    train_segs, val_segs, test_segs = segments[:train_end], segments[train_end:val_end], segments[val_end:]

    if not train_segs or not val_segs or not test_segs:
        raise ValueError("Train, validation, and test splits must all contain at least one segment.")

    X_train, y_train, I_train, s_train = generate_samples(
        train_segs,
        seq_cfg["input_steps"],
        seq_cfg["output_steps"],
        feature_scaler,
        target_scaler,
        ds_cfg["feature_columns"],
        ds_cfg["target_columns"],
        ds_cfg["current_column"],
        ds_cfg["soc_column"],
        step=1,
    )
    X_val, y_val, I_val, s_val = generate_samples(
        val_segs,
        seq_cfg["input_steps"],
        seq_cfg["output_steps"],
        feature_scaler,
        target_scaler,
        ds_cfg["feature_columns"],
        ds_cfg["target_columns"],
        ds_cfg["current_column"],
        ds_cfg["soc_column"],
        step=1,
    )
    X_test, y_test, I_test, s_test = generate_samples(
        test_segs,
        seq_cfg["input_steps"],
        seq_cfg["output_steps"],
        feature_scaler,
        target_scaler,
        ds_cfg["feature_columns"],
        ds_cfg["target_columns"],
        ds_cfg["current_column"],
        ds_cfg["soc_column"],
        step=data_cfg["test_step"],
    )

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(SeqDataset(X_train, y_train, I_train, s_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val, I_val, s_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SeqDataset(X_test, y_test, I_test, s_test), batch_size=batch_size, shuffle=False)

    return {
        "df": df,
        "segments": segments,
        "train_segments": train_segs,
        "val_segments": val_segs,
        "test_segments": test_segs,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_arrays": (X_train, y_train, I_train, s_train),
        "val_arrays": (X_val, y_val, I_val, s_val),
        "test_arrays": (X_test, y_test, I_test, s_test),
        "min_len": min_len,
        "timestamps": build_timestamp_index(test_segs, data_cfg["time_column"], min_len, seq_cfg["input_steps"], seq_cfg["output_steps"]),
    }


def save_scalers(feature_scaler, target_scaler, scaler_dir: str | Path, experiment_name: str, in_steps: int, out_steps: int) -> tuple[Path, Path]:
    scaler_dir = Path(scaler_dir)
    scaler_dir.mkdir(parents=True, exist_ok=True)
    feature_path = scaler_dir / f"feature_scaler_{experiment_name}_in{in_steps}_out{out_steps}.pkl"
    target_path = scaler_dir / f"target_scaler_{experiment_name}_in{in_steps}_out{out_steps}.pkl"
    joblib.dump(feature_scaler, feature_path)
    joblib.dump(target_scaler, target_path)
    return feature_path, target_path
