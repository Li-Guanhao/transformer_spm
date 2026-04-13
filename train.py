from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_preprocessing import load_config, prepare_data, resolve_path, save_scalers
from model import TransformerForecaster
from model.spm_constraint import apply_physical_correction, calculate_p2d_constraints_scaled, mean_relative_error, physical_constrained_loss


def ensure_directories(cfg: dict):
    for key in ["output_dir", "model_dir", "prediction_dir", "scaler_dir"]:
        Path(resolve_path(cfg["paths"][key])).mkdir(parents=True, exist_ok=True)


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


def run_experiment(cfg: dict, data_bundle: dict, experiment: dict, device: torch.device, now1: datetime):
    seq_cfg = cfg["sequence"]
    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    test_loader = data_bundle["test_loader"]
    feature_scaler = data_bundle["feature_scaler"]
    target_scaler = data_bundle["target_scaler"]
    timestamps = data_bundle["timestamps"]
    test_arrays = data_bundle["test_arrays"]

    model = build_model(cfg, use_pe=experiment["use_pe"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, min_lr=1e-6)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    use_amp = bool(cfg["training"].get("use_amp", True)) and device.type == "cuda"

    best_val = np.inf
    patience_counter = 0
    model_dir = Path(resolve_path(cfg["paths"]["model_dir"]))
    pred_dir = Path(resolve_path(cfg["paths"]["prediction_dir"]))
    scaler_dir = Path(resolve_path(cfg["paths"]["scaler_dir"]))
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    in_steps = seq_cfg["input_steps"]
    out_steps = seq_cfg["output_steps"]
    model_save_path = model_dir / f"best_model_{experiment['name']}_in{in_steps}_out{out_steps}.pth"

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for xb, yb, ib, sb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            if experiment["use_physical"]:
                p2d_targets_scaled = calculate_p2d_constraints_scaled(sb, ib, target_scaler, cfg, device)
            else:
                p2d_targets_scaled = {"p2d_pred_scaled": torch.zeros(1, device=device)}
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xb)
                loss = physical_constrained_loss(pred, yb, p2d_targets_scaled, cfg, use_physical=experiment["use_physical"])
            scaler_amp.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_sum += loss.item()
            train_count += 1

        train_loss = train_sum / max(1, train_count)

        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb, ib, sb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                if experiment["use_physical"]:
                    p2d_targets_scaled = calculate_p2d_constraints_scaled(sb, ib, target_scaler, cfg, device)
                else:
                    p2d_targets_scaled = {"p2d_pred_scaled": torch.zeros(1, device=device)}
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(xb)
                    val_loss_batch = physical_constrained_loss(pred, yb, p2d_targets_scaled, cfg, use_physical=experiment["use_physical"])
                val_sum += val_loss_batch.item()
                val_count += 1

        val_loss = val_sum / max(1, val_count)
        scheduler.step(val_loss)
        print(f"[{experiment['name']}] Epoch {epoch:3d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"[{experiment['name']}] Saved best checkpoint to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["patience"]:
                print(f"[{experiment['name']}] Early stopping triggered.")
                break

    print(f"[{experiment['name']}] Loading best checkpoint for evaluation...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    y_pred_list = []
    all_X_test = []
    all_I_test = []
    with torch.no_grad():
        for xb, _, ib, _ in tqdm(test_loader, desc=f"Testing-{experiment['name']}"):
            y_pred_list.append(model(xb.to(device)).cpu())
            all_X_test.append(xb.cpu().numpy())
            all_I_test.append(ib.cpu().numpy())

    if not y_pred_list:
        print(f"[{experiment['name']}] No test samples were found.")
        return

    y_pred_scaled = torch.cat(y_pred_list, dim=0).numpy()
    X_test_all_batches = np.concatenate(all_X_test, axis=0)
    I_test_all_batches = np.concatenate(all_I_test, axis=0)

    true_test_flat = target_scaler.inverse_transform(pd.DataFrame(data_bundle["test_arrays"][1].reshape(-1, len(cfg["dataset"]["target_columns"])), columns=cfg["dataset"]["target_columns"]))
    pred_flat_corrected = apply_physical_correction(y_pred_scaled, X_test_all_batches, I_test_all_batches, feature_scaler, target_scaler, cfg)

    result = pd.DataFrame({"Timestamp": timestamps})
    for i, col in enumerate(cfg["dataset"]["target_columns"]):
        result[f"TrueValue_{col}"] = true_test_flat[:, i]
        result[f"CorrectedPrediction_{col}"] = pred_flat_corrected[:, i]

    csv_path = pred_dir / f"predictions_{experiment['name']}_transformer_spm_constrained_loss_{in_steps}_{out_steps}.csv"
    result.to_csv(csv_path, index=False, encoding=cfg["data"]["encoding"])
    print(f"[{experiment['name']}] Saved predictions to {csv_path}")

    print(f"\n[{experiment['name']}] Metrics:")
    metrics = {}
    for i, col in enumerate(cfg["dataset"]["target_columns"]):
        rmse = np.sqrt(np.mean((true_test_flat[:, i] - pred_flat_corrected[:, i]) ** 2))
        mre = mean_relative_error(true_test_flat[:, i], pred_flat_corrected[:, i])
        metrics[col] = {"rmse": float(rmse), "mre": float(mre)}
        print(f"[{experiment['name']}] RMSE {col}: {rmse:.6f}")
        print(f"[{experiment['name']}] MRE  {col}: {mre:.6f}")

    save_scalers(feature_scaler, target_scaler, scaler_dir, experiment["name"], in_steps, out_steps)
    print(f"[{experiment['name']}] Saved scalers.")
    print(f"[{experiment['name']}] Elapsed time: {datetime.now() - now1}")
    return {"model_path": model_save_path, "prediction_path": csv_path, "metrics": metrics}


def main():
    cfg = load_config()
    ensure_directories(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now1 = datetime.now()
    print(f"Start time: {now1}")
    print(f"Device: {device}")

    data_bundle = prepare_data(cfg)
    print(f"Segments: {len(data_bundle['segments'])} | Test samples: {len(data_bundle['test_arrays'][0])}")

    for experiment in cfg["experiments"]:
        print("\n" + "=" * 30)
        print(f"Experiment: {experiment['name']} | use_pe={experiment['use_pe']} | use_physical={experiment['use_physical']}")
        print("=" * 30)
        run_experiment(cfg, data_bundle, experiment, device, now1)

    print("All experiments completed.")
    print("Total elapsed:", datetime.now() - now1)


if __name__ == "__main__":
    main()
