from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_preprocessing import load_config, prepare_data, resolve_path
from model import TransformerForecaster
from model.spm_constraint import apply_physical_correction, mean_relative_error


def build_model(cfg: dict, use_pe: bool):
    model_cfg = cfg["model"]
    seq_cfg = cfg["sequence"]
    ds_cfg = cfg["dataset"]
    return TransformerForecaster(
        input_dim=len(ds_cfg["feature_columns"]),
        d_model=model_cfg["d_model"],
        nhead=model_cfg["n_head"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        out_steps=seq_cfg["output_steps"],
        n_targets=len(ds_cfg["target_columns"]),
        max_len=model_cfg["max_seq_len"],
        use_pe=use_pe,
    )


def evaluate_experiment(cfg: dict, data_bundle: dict, experiment: dict, device: torch.device):
    seq_cfg = cfg["sequence"]
    pred_dir = Path(resolve_path(cfg["paths"]["prediction_dir"]))
    model_dir = Path(resolve_path(cfg["paths"]["model_dir"]))
    scaler_dir = Path(resolve_path(cfg["paths"]["scaler_dir"]))
    pred_dir.mkdir(parents=True, exist_ok=True)

    in_steps = seq_cfg["input_steps"]
    out_steps = seq_cfg["output_steps"]
    model_path = model_dir / f"best_model_{experiment['name']}_in{in_steps}_out{out_steps}.pth"
    feature_scaler_path = scaler_dir / f"feature_scaler_{experiment['name']}_in{in_steps}_out{out_steps}.pkl"
    target_scaler_path = scaler_dir / f"target_scaler_{experiment['name']}_in{in_steps}_out{out_steps}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not feature_scaler_path.exists() or not target_scaler_path.exists():
        raise FileNotFoundError("Missing scaler files for evaluation.")

    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)

    model = build_model(cfg, use_pe=experiment["use_pe"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader = data_bundle["test_loader"]
    timestamps = data_bundle["timestamps"]
    test_arrays = data_bundle["test_arrays"]

    y_pred_list = []
    all_X_test = []
    all_I_test = []
    with torch.no_grad():
        for xb, _, ib, _ in tqdm(test_loader, desc=f"Evaluating-{experiment['name']}"):
            y_pred_list.append(model(xb.to(device)).cpu())
            all_X_test.append(xb.cpu().numpy())
            all_I_test.append(ib.cpu().numpy())

    if not y_pred_list:
        raise ValueError("No test samples were found.")

    y_pred_scaled = torch.cat(y_pred_list, dim=0).numpy()
    X_test_all_batches = np.concatenate(all_X_test, axis=0)
    I_test_all_batches = np.concatenate(all_I_test, axis=0)
    pred_flat_corrected = apply_physical_correction(y_pred_scaled, X_test_all_batches, I_test_all_batches, feature_scaler, target_scaler, cfg)
    true_test_flat = target_scaler.inverse_transform(
        pd.DataFrame(test_arrays[1].reshape(-1, len(cfg["dataset"]["target_columns"])), columns=cfg["dataset"]["target_columns"])
    )

    result = pd.DataFrame({"Timestamp": timestamps})
    for i, col in enumerate(cfg["dataset"]["target_columns"]):
        result[f"TrueValue_{col}"] = true_test_flat[:, i]
        result[f"CorrectedPrediction_{col}"] = pred_flat_corrected[:, i]

    csv_path = pred_dir / f"evaluation_{experiment['name']}_in{in_steps}_out{out_steps}.csv"
    result.to_csv(csv_path, index=False, encoding=cfg["data"]["encoding"])

    metrics = {}
    for i, col in enumerate(cfg["dataset"]["target_columns"]):
        rmse = np.sqrt(np.mean((true_test_flat[:, i] - pred_flat_corrected[:, i]) ** 2))
        mre = mean_relative_error(true_test_flat[:, i], pred_flat_corrected[:, i])
        metrics[col] = {"rmse": float(rmse), "mre": float(mre)}
        print(f"[{experiment['name']}] RMSE {col}: {rmse:.6f}")
        print(f"[{experiment['name']}] MRE  {col}: {mre:.6f}")

    print(f"[{experiment['name']}] Saved evaluation results to {csv_path}")
    return {"csv_path": csv_path, "metrics": metrics}


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_bundle = prepare_data(cfg)

    for experiment in cfg["experiments"]:
        evaluate_experiment(cfg, data_bundle, experiment, device)


if __name__ == "__main__":
    main()
