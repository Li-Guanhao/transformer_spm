from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch


def mean_relative_error(true, pred, epsilon: float = 1e-8):
    return np.mean(np.abs((true - pred) / (true + epsilon)))


def ocv_poly(soc, coeffs):
    return np.polyval(np.asarray(coeffs), soc)


def physical_constrained_loss(pred_scaled, true_scaled, p2d_output_scaled, cfg: dict[str, Any], use_physical: bool = True):
    mae_loss = torch.mean(torch.abs(pred_scaled - true_scaled))
    if not use_physical:
        return mae_loss

    target_columns = cfg["dataset"]["target_columns"]
    v_target = cfg["dataset"]["voltage_target"]
    soc_target = cfg["dataset"]["soc_target"]
    v_idx_in_targets = target_columns.index(v_target)
    soc_idx_in_targets = target_columns.index(soc_target)

    p2d_pred_scaled = p2d_output_scaled["p2d_pred_scaled"]
    v_pred_scaled = pred_scaled[:, :, v_idx_in_targets]
    v_p2d_scaled = p2d_pred_scaled[:, :, v_idx_in_targets]
    voltage_constraint = torch.mean(torch.abs(v_pred_scaled - v_p2d_scaled))

    soc_pred_scaled = pred_scaled[:, :, soc_idx_in_targets]
    soc_p2d_scaled = p2d_pred_scaled[:, :, soc_idx_in_targets]
    soc_constraint = torch.mean(torch.abs(soc_pred_scaled - soc_p2d_scaled))

    return mae_loss + cfg["physics"]["w_p2d_voltage"] * voltage_constraint + cfg["physics"]["w_coulomb_soc"] * soc_constraint


def calculate_p2d_constraints_scaled(soc_start_unscaled, i_future_unscaled, scaler, cfg: dict[str, Any], device):
    physics_cfg = cfg["physics"]
    target_columns = cfg["dataset"]["target_columns"]
    v_target = cfg["dataset"]["voltage_target"]
    soc_target = cfg["dataset"]["soc_target"]
    v_idx_in_targets = target_columns.index(v_target)
    soc_idx_in_targets = target_columns.index(soc_target)

    c_nominal_as = physics_cfg["nominal_capacity_ah"] * 3600.0
    soc_start_np = soc_start_unscaled.cpu().numpy()
    i_future_np = i_future_unscaled.cpu().numpy()

    batch_size, n_steps = i_future_np.shape
    soc_bulk = np.zeros((batch_size, n_steps + 1), dtype=np.float64)
    soc_surface = np.zeros((batch_size, n_steps + 1), dtype=np.float64)

    soc_bulk[:, 0] = soc_start_np
    soc_surface[:, 0] = soc_start_np

    for t in range(n_steps):
        delta_soc = (i_future_np[:, t] * physics_cfg["delta_t_s"]) / c_nominal_as
        soc_bulk[:, t + 1] = soc_bulk[:, t] - delta_soc
        soc_surface[:, t + 1] = soc_surface[:, t] - delta_soc + (soc_bulk[:, t] - soc_surface[:, t]) * (
            physics_cfg["delta_t_s"] / physics_cfg["tau_diffusion"]
        )

    soc_bulk = np.clip(soc_bulk, 0.0, 1.0)
    soc_surface = np.clip(soc_surface, 0.0, 1.0)

    future_soc_surface = soc_surface[:, :-1]
    u_eq = ocv_poly(future_soc_surface, physics_cfg["ocv_coeffs"])
    r_ct = physics_cfg["r_ct_base"] + physics_cfg["r_ct_ext"] * (future_soc_surface - 0.5) ** 2
    eta_overpotential = i_future_np * r_ct
    ohmic_drop = i_future_np * physics_cfg["r_ohmic"]
    v_p2d_unscaled = u_eq - eta_overpotential - ohmic_drop
    soc_p2d_unscaled = soc_bulk[:, 1:]

    p2d_df_unscaled = pd.DataFrame(0.0, index=np.arange(v_p2d_unscaled.size), columns=target_columns)
    p2d_df_unscaled.iloc[:, v_idx_in_targets] = v_p2d_unscaled.flatten()
    p2d_df_unscaled.iloc[:, soc_idx_in_targets] = soc_p2d_unscaled.flatten()

    p2d_scaled = scaler.transform(p2d_df_unscaled).reshape(batch_size, n_steps, len(target_columns))
    return {"p2d_pred_scaled": torch.from_numpy(p2d_scaled).float().to(device)}


def apply_physical_correction(y_pred_scaled, x_test_all, i_future_all, feature_scaler, target_scaler, cfg: dict[str, Any]):
    target_columns = cfg["dataset"]["target_columns"]
    feature_columns = cfg["dataset"]["feature_columns"]
    v_target = cfg["dataset"]["voltage_target"]
    soc_target = cfg["dataset"]["soc_target"]
    v_idx_in_targets = target_columns.index(v_target)
    soc_idx_in_targets = target_columns.index(soc_target)
    soc_idx_in_features = feature_columns.index(soc_target)

    pred_df_scaled = pd.DataFrame(y_pred_scaled.reshape(-1, len(target_columns)), columns=target_columns)
    pred_unscaled = target_scaler.inverse_transform(pred_df_scaled).reshape(y_pred_scaled.shape)

    last_step_features_scaled = x_test_all[:, -1, :]
    last_step_df_scaled = pd.DataFrame(last_step_features_scaled, columns=feature_columns)
    soc_start_unscaled = feature_scaler.inverse_transform(last_step_df_scaled)[:, soc_idx_in_features]

    c_nominal_as = cfg["physics"]["nominal_capacity_ah"] * 3600.0
    delta_soc = np.cumsum(i_future_all * cfg["physics"]["delta_t_s"], axis=1) / c_nominal_as
    corrected_soc = soc_start_unscaled[:, np.newaxis] - delta_soc

    pred_unscaled[:, :, soc_idx_in_targets] = corrected_soc
    pred_unscaled[:, :, v_idx_in_targets] = np.clip(
        pred_unscaled[:, :, v_idx_in_targets],
        cfg["physics"]["v_min_limit"],
        cfg["physics"]["v_max_limit"],
    )
    return pred_unscaled.reshape(-1, len(target_columns))
