# -*- coding: utf-8 -*-
"""
End-to-end time-series forecasting script (with SPM-like physical constraints in the loss function) + ablation experiments
========================================================================================================================
Data:  /whut_data/wjy/chiao/battery/battery_code_self/data/LK6AEAE47RB443988_change.CSV
GPU : 3 (24 GB RTX3090)
Task: Predict future sequences from historical sequences and apply physical constraints during training
Output: predictions_*.csv + best_model_*.pth + scaler_*.pkl

Key features
------------
* Sliding-window within each segment; automatically removes discontinuous segments with gaps > 5 min
* ***MODIFIED V3.0: Replaced Bi-LSTM + Temporal Attention pooling with Transformer + Positional Encoding***
* ***MODIFIED V2.0: Integrated a more realistic single-particle physical model (SPM-like) as a soft constraint into the loss function.***
* AMP (automatic mixed precision), gradient clipping, dynamic learning-rate scheduling, early stopping
* Fixed: resolved Scikit-learn feature names UserWarning and pandas dtype FutureWarning
* Added: ablation experiments:
    a) remove positional encoding (PE)
    b) remove SPM electrochemical physical constraint
* Launch command: nohup python -u transformer_spm_ablation.py >> transformer_spm_ablation.out 2>&1 &
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datetime import datetime
import math # for positional encoding

# ---------------------------- Global configuration ----------------------------
DATA_FILE = Path("/data/liguanhao/battery_code_self_lgh/data/csv_add/add10.csv")
OUTPUT_DIR = DATA_FILE.parent
BATCH_SIZE = 1024
TIME_STEPS_IN = 360      # Input sequence length
TIME_STEPS_OUT = 2       # Output sequence length
TRAIN_RATIO, VAL_RATIO = 0.7, 0.2  # Dataset split ratio
EPOCHS = 100             # Maximum number of training epochs
PATIENCE = 10            # Early-stopping patience
LR = 1e-3                # Initial learning rate
WEIGHT_DECAY = 1e-4      # Weight decay

# --- Transformer model parameters ---
D_MODEL = 256            # Transformer feature dimension (hidden_dim)
N_HEAD = 8               # Number of multi-head attention heads
NUM_ENCODER_LAYERS = 3   # Number of Transformer encoder layers
DIM_FEEDFORWARD = 512    # Feed-forward network dimension, usually 2 to 4 times D_MODEL
DROPOUT = 0.25           # Dropout rate
MAX_SEQ_LEN = TIME_STEPS_IN + 100 # Maximum sequence length for positional encoding, slightly larger than TIME_STEPS_IN

# ==================== Physical-model and loss-function parameters ====================
C_NOMINAL_AH = 135.0     # Rated battery capacity (Ah)
DELTA_T_S = 2.0          # Data sampling interval (seconds)
V_MIN_LIMIT = 2.5        # Lower hard clip limit for final predicted voltage
V_MAX_LIMIT = 3.65       # Upper hard clip limit for final predicted voltage
W_P2D_VOLTAGE = 0.1      # Weight of the physical-model voltage constraint term
W_COULOMB_SOC = 0.05     # Weight of the Coulomb-counting SOC constraint term

# --- Added: Single-particle model (SPM-like) physical parameters ---
R_OHMIC = 0.0015  # Ohms (Ω), battery ohmic internal resistance (current collectors, electrolyte, etc.)
R_CT_BASE = 0.006 # Ohms (Ω), base charge-transfer resistance
R_CT_EXT = 0.1    # Ohms (Ω), additional resistance coefficient at SOC extremes
TAU_DIFFUSION = 1800.0 # Seconds (s), solid-phase diffusion time constant

# Polynomial coefficients for the OCV-SOC curve (fit from experimental data)
OCV_COEFFS = np.array([-0.5, 1.2, -0.8, 0.1, 3.25])

# ==================== Mode control switches ====================
GENERALIZATION_TEST_ONLY = False # Whether to run only generalization testing
EXTERNAL_TEST_FILE = Path("/whut_data/wjy/chiao/battery/battery_code_self/data/csv_processed/LK6AEAE4XRB440390_change.CSV")

# ---------------------------- Models and helper functions ----------------------------

# 1. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0), :]

# 2. Transformer Forecaster (supports optional positional encoding)
class TransformerForecaster(nn.Module):
    def __init__(self,
                 input_dim,
                 d_model,              # Transformer feature dimension
                 nhead,                # Number of MultiheadAttention heads
                 num_encoder_layers,   # Number of TransformerEncoderLayer blocks
                 dim_feedforward,      # Feed-forward network dimension
                 dropout,
                 out_steps,
                 n_targets,
                 max_len=5000,
                 use_pe=True):         # Added: whether to use positional encoding
        super().__init__()

        self.d_model = d_model
        self.out_steps = out_steps
        self.n_targets = n_targets
        self.use_pe = use_pe

        # 1. Input projection layer: maps input features to the Transformer d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 2. Positional encoding (optional)
        if use_pe:
            self.positional_encoding = PositionalEncoding(d_model, max_len)
        else:
            self.positional_encoding = None

        # 3. Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False # PyTorch Transformer defaults to (seq_len, batch_size, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # 4. Output prediction layer
        self.output_projection = nn.Linear(d_model, out_steps * n_targets)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)

        # 1. Project to d_model dimension
        x = self.input_projection(x) # Shape: (batch_size, seq_len, d_model)

        # 2. PyTorch Transformer expects (seq_len, batch_size, features)
        x = x.permute(1, 0, 2) # Shape: (seq_len, batch_size, d_model)

        # 3. Add positional encoding if enabled
        if self.positional_encoding is not None:
            x = self.positional_encoding(x) # Shape: (seq_len, batch_size, d_model)

        # 4. Pass through the Transformer encoder
        transformer_output = self.transformer_encoder(x) # Shape: (seq_len, batch_size, d_model)

        # 5. Pooling: use the mean over all time steps as the context vector
        context_vector = transformer_output.mean(dim=0) # Shape: (batch_size, d_model)

        # 6. Predict future sequences
        out = self.output_projection(context_vector) # Shape: (batch_size, out_steps * n_targets)
        
        return out.view(-1, self.out_steps, self.n_targets) # Reshape to (batch_size, out_steps, n_targets)

def generate_samples(segment_list, in_steps, out_steps, step=1):
    X, y, I_future, soc_start = [], [], [], []
    for seg in segment_list:
        f_scaled = feature_scaler.transform(seg[feature_columns])
        t_scaled = target_scaler.transform(seg[target_columns])
        currents_unscaled = seg["总电流"].values
        soc_unscaled = seg["SOC"].values
        for i in range(0, len(seg) - in_steps - out_steps + 1, step):
            X.append(f_scaled[i:i+in_steps])
            y.append(t_scaled[i+in_steps:i+in_steps+out_steps])
            I_future.append(currents_unscaled[i+in_steps:i+in_steps+out_steps])
            soc_start.append(soc_unscaled[i+in_steps-1])
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32),
            np.asarray(I_future, dtype=np.float32), np.asarray(soc_start, dtype=np.float32))

def mean_relative_error(true, pred, epsilon=1e-8):
    return np.mean(np.abs((true - pred) / (true + epsilon)))

class SeqDataset(Dataset):
    def __init__(self, X, y, I_future, soc_start):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.I_future = torch.from_numpy(I_future)
        self.soc_start = torch.from_numpy(soc_start)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.I_future[idx], self.soc_start[idx]

# ---------------------------- Environment and device ----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
now1 = datetime.now()
print(f"Starting: Transformer model with integrated SPM-like physical constraints (with ablation experiments), {now1}")
print("Device:", device)

# ---------------------------- Loading and segmentation ----------------------------
print("Loading data...")
if not GENERALIZATION_TEST_ONLY:
    df = pd.read_csv(DATA_FILE, encoding="gbk")
else:
    df = pd.read_csv(EXTERNAL_TEST_FILE, encoding="gbk")
df["数据采集时间"] = pd.to_datetime(df["数据采集时间"], format="%Y-%m-%d %H:%M:%S")
seg_flags = df["数据采集时间"].diff().dt.total_seconds().fillna(0).gt(300)
df["seg_id"] = seg_flags.cumsum()
MIN_LEN = TIME_STEPS_IN + TIME_STEPS_OUT
segments = [g.reset_index(drop=True) for _, g in df.groupby("seg_id") if len(g) >= MIN_LEN]
print(f"Valid segment count: {len(segments)}  | Total sample points: {sum(len(s) for s in segments):,}")

# ---------------------------- Feature and target columns ----------------------------
feature_columns = ["Cell voltage", "Cell temperature", "Total current", "Total voltage", "SOC"]
target_columns = ["Cell voltage", "Cell temperature", "SOC"]
v_idx_in_targets = target_columns.index("Cell voltage")
soc_idx_in_targets = target_columns.index("SOC")

# ==================== Physical constraint loss function ====================
def ocv_poly(soc):
    """Compute open-circuit voltage from SOC (vectorized)."""
    return np.polyval(OCV_COEFFS, soc)

def physical_constrained_loss(pred_scaled, true_scaled, p2d_output_scaled, w_volt, w_soc, use_physical=True):
    """
    Compute the hybrid loss with physical constraints.
    If use_physical=False, return only the MAE main loss (used for the no_spm ablation).
    """
    # Main loss: mean absolute error between predictions and ground truth
    mae_loss = torch.mean(torch.abs(pred_scaled - true_scaled))
    if not use_physical:
        return mae_loss
    p2d_pred_scaled = p2d_output_scaled["p2d_pred_scaled"]

    # Voltage constraint: penalize deviation from the SPM physical-model voltage prediction
    v_pred_scaled = pred_scaled[:, :, v_idx_in_targets]
    v_p2d_scaled = p2d_pred_scaled[:, :, v_idx_in_targets]
    voltage_constraint = torch.mean(torch.abs(v_pred_scaled - v_p2d_scaled))

    # SOC constraint: penalize deviation from Coulomb-counting SOC
    soc_pred_scaled = pred_scaled[:, :, soc_idx_in_targets]
    soc_p2d_scaled = p2d_pred_scaled[:, :, soc_idx_in_targets]
    soc_constraint = torch.mean(torch.abs(soc_pred_scaled - soc_p2d_scaled))
    
    # Total loss = main loss + weighted physical-constraint losses
    total_loss = mae_loss + w_volt * voltage_constraint + w_soc * soc_constraint
    return total_loss

def calculate_p2d_constraints_scaled(soc_start_unscaled, i_future_unscaled, scaler):
    """
    Compute the physical-constraint targets using a simplified single-particle model (SPM-like).
    1. Compute bulk SOC via Coulomb counting.
    2. Approximate particle surface SOC via a first-order ordinary differential equation (ODE).
    3. Compute terminal voltage based on surface SOC and reaction kinetics:
       V_terminal = U_eq(soc_surface) - η_overpotential - I * R_ohmic
    4. Normalize the computed voltage and SOC sequences and use them as soft targets in the loss function.
    """
    C_NOMINAL_AS = C_NOMINAL_AH * 3600.0
    soc_start_np = soc_start_unscaled.cpu().numpy()
    i_future_np = i_future_unscaled.cpu().numpy()
    
    batch_size, n_steps = i_future_np.shape
    
    # Initialize state arrays
    soc_bulk = np.zeros((batch_size, n_steps + 1))
    soc_surface = np.zeros((batch_size, n_steps + 1))
    
    # Set initial conditions (t=0); assume surface SOC equals bulk SOC initially
    soc_bulk[:, 0] = soc_start_np
    soc_surface[:, 0] = soc_start_np
    
    # Simulate across the batch step by step
    for t in range(n_steps):
        # Step 1: update bulk SOC (based on Coulomb counting)
        delta_soc = (i_future_np[:, t] * DELTA_T_S) / C_NOMINAL_AS
        soc_bulk[:, t+1] = soc_bulk[:, t] - delta_soc
        
        # Step 2: update surface SOC (accounting for current consumption and solid-phase diffusion)
        # Discretized first-order ODE: d(soc_s)/dt ≈ (soc_b - soc_s)/τ_diff
        soc_surface[:, t+1] = soc_surface[:, t] - delta_soc + \
                              (soc_bulk[:, t] - soc_surface[:, t]) * (DELTA_T_S / TAU_DIFFUSION)

    # Clip SOC values to ensure they remain within the physical range [0, 1]
    soc_bulk = np.clip(soc_bulk, 0.0, 1.0)
    soc_surface = np.clip(soc_surface, 0.0, 1.0)
    
    # Step 3: compute the terminal-voltage sequence predicted by the SPM
    future_soc_surface = soc_surface[:, :-1]
    
    # 3a. Compute the surface equilibrium potential U_eq = OCV(soc_surface)
    u_eq = ocv_poly(future_soc_surface)
    
    # 3b. Compute surface overpotential η_s, approximated as I * R_ct(soc_surface)
    r_ct = R_CT_BASE + R_CT_EXT * (future_soc_surface - 0.5)**2
    eta_overpotential = i_future_np * r_ct
    
    # 3c. Compute the total ohmic drop I * R_ohmic
    ohmic_drop = i_future_np * R_OHMIC
    
    # 3d. Compute the final terminal voltage
    v_p2d_unscaled = u_eq - eta_overpotential - ohmic_drop
    
    # Use the Coulomb-counting result directly as the physical-model SOC prediction
    soc_p2d_unscaled = soc_bulk[:, 1:]

    # Step 4: normalize the physical-model voltage and SOC sequences
    p2d_df_unscaled = pd.DataFrame(0.0, index=np.arange(v_p2d_unscaled.size), columns=target_columns)
    p2d_df_unscaled.iloc[:, v_idx_in_targets] = v_p2d_unscaled.flatten()
    p2d_df_unscaled.iloc[:, soc_idx_in_targets] = soc_p2d_unscaled.flatten()

    p2d_scaled = scaler.transform(p2d_df_unscaled).reshape(batch_size, n_steps, len(target_columns))

    return {
        "p2d_pred_scaled": torch.from_numpy(p2d_scaled).float().to(device),
    }

# ---------------------------- Data preprocessing: normalization and sliding windows (done once, shared) ----------------------------
if not GENERALIZATION_TEST_ONLY:
    print("Fitting normalization scalers...")
    all_feat_df = pd.concat([seg[feature_columns] for seg in segments], ignore_index=True)
    all_tgt_df  = pd.concat([seg[target_columns] for seg in segments], ignore_index=True)
    feature_scaler = MinMaxScaler().fit(all_feat_df)
    target_scaler  = MinMaxScaler().fit(all_tgt_df)

    n_total = len(segments)
    train_end, val_end = int(TRAIN_RATIO * n_total), int(TRAIN_RATIO * n_total) + int(VAL_RATIO * n_total)
    train_segs, val_segs, test_segs = segments[:train_end], segments[train_end:val_end], segments[val_end:]

    print("Generating sliding-window samples...")
    X_train, y_train, I_train, s_train = generate_samples(train_segs, TIME_STEPS_IN, TIME_STEPS_OUT, step=1)
    X_val, y_val, I_val, s_val         = generate_samples(val_segs, TIME_STEPS_IN, TIME_STEPS_OUT, step=1)
    X_test, y_test, I_test, s_test     = generate_samples(test_segs, TIME_STEPS_IN, TIME_STEPS_OUT, step=TIME_STEPS_OUT)
    print(f"Train set: {len(X_train)} | Validation set: {len(X_val)} | Test set: {len(X_test)}")

    train_loader = DataLoader(SeqDataset(X_train, y_train, I_train, s_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(SeqDataset(X_val, y_val, I_val, s_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(SeqDataset(X_test, y_test, I_test, s_test), batch_size=BATCH_SIZE, shuffle=False)
else:
    print("Generalization test mode is not implemented yet (same as the original script).")

# ---------------------------- Training and evaluation function (shared by all experiment configs) ----------------------------
def train_and_evaluate(experiment_name, use_pe=True, use_physical=True):
    """
    Run a complete training/validation/testing workflow and save the model and prediction results.
    experiment_name: used for file naming
    use_pe: whether to use positional encoding in the model
    use_physical: whether to include the SPM physical constraint in the loss function
    """
    print("\n" + "="*30)
    print(f"Starting experiment: {experiment_name} | use_pe={use_pe} | use_physical={use_physical}")
    print("="*30)

    model = TransformerForecaster(
        input_dim=len(feature_columns),
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        out_steps=TIME_STEPS_OUT,
        n_targets=len(target_columns),
        max_len=MAX_SEQ_LEN,
        use_pe=use_pe
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6, verbose=True)
    scaler_amp = torch.cuda.amp.GradScaler() # Used for automatic mixed-precision training

    best_val, pat_cnt = np.inf, 0
    model_save_path = OUTPUT_DIR/f"best_model_{experiment_name}_in{TIME_STEPS_IN}_out{TIME_STEPS_OUT}.pth"

    for epoch in range(1, EPOCHS+1):
        model.train()
        train_sum, train_count = 0.0, 0
        for xb, yb, ib, sb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            p2d_targets_scaled = calculate_p2d_constraints_scaled(sb, ib, target_scaler) if use_physical else {"p2d_pred_scaled": torch.zeros(1).to(device)}
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(xb)
                loss = physical_constrained_loss(pred, yb, p2d_targets_scaled, W_P2D_VOLTAGE, W_COULOMB_SOC, use_physical=use_physical)
            scaler_amp.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_sum += loss.item()
            train_count += 1
        train_loss = train_sum / max(1, train_count)

        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for xb, yb, ib, sb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                p2d_targets_scaled = calculate_p2d_constraints_scaled(sb, ib, target_scaler) if use_physical else {"p2d_pred_scaled": torch.zeros(1).to(device)}
                pred = model(xb)
                val_loss_batch = physical_constrained_loss(pred, yb, p2d_targets_scaled, W_P2D_VOLTAGE, W_COULOMB_SOC, use_physical=use_physical)
                val_sum += val_loss_batch.item()
                val_count += 1
        val_loss = val_sum / max(1, val_count)
        scheduler.step(val_loss)
        print(f"[{experiment_name}] Epoch {epoch:2d} | Train loss {train_loss: .6f} | Val loss {val_loss: .6f}")

        if val_loss + 1e-8 < best_val:
            best_val, pat_cnt = val_loss, 0
            torch.save(model.state_dict(), model_save_path)
            print(f"[{experiment_name}] Saved new best model -> {model_save_path}")
        else:
            pat_cnt += 1
            if pat_cnt >= PATIENCE:
                print(f"[{experiment_name}] Early stopping triggered; training stopped."); break

    # ---------------------------- Testing & saving results ----------------------------
    print(f"[{experiment_name}] Loading best model for testing...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    y_pred_list = []
    all_X_test = []
    all_I_test = []
    with torch.no_grad():
        for xb, _, ib, _ in tqdm(test_loader, desc=f"Testing-{experiment_name}"):
            y_pred_list.append(model(xb.to(device)).cpu())
            all_X_test.append(xb.cpu().numpy())
            all_I_test.append(ib.cpu().numpy())

    if len(y_pred_list) == 0:
        print(f"[{experiment_name}] No test samples were found; skipping evaluation.")
        return

    y_pred_scaled = torch.cat(y_pred_list, dim=0).numpy()
    X_test_all_batches = np.concatenate(all_X_test, axis=0)
    I_test_all_batches = np.concatenate(all_I_test, axis=0)

    # Inverse-normalize and apply physical hard-constraint correction (apply_physical_correction stays consistent with the original script)
    def apply_physical_correction(y_pred_scaled, x_test_all, i_future_all):
        pred_df_scaled = pd.DataFrame(y_pred_scaled.reshape(-1, len(target_columns)), columns=target_columns)
        pred_unscaled = target_scaler.inverse_transform(pred_df_scaled).reshape(y_pred_scaled.shape)

        soc_idx_in_features = feature_columns.index("SOC")
        last_step_features_scaled = x_test_all[:, -1, :] # Get the features at the last time step of the input sequence
        last_step_df_scaled = pd.DataFrame(last_step_features_scaled, columns=feature_columns)
        soc_start_unscaled = feature_scaler.inverse_transform(last_step_df_scaled)[:, soc_idx_in_features]

        C_NOMINAL_AS = C_NOMINAL_AH * 3600.0
        delta_soc = np.cumsum(i_future_all * DELTA_T_S, axis=1) / C_NOMINAL_AS
        corrected_soc = soc_start_unscaled[:, np.newaxis] - delta_soc

        pred_unscaled[:, :, soc_idx_in_targets] = corrected_soc
        pred_unscaled[:, :, v_idx_in_targets] = np.clip(pred_unscaled[:, :, v_idx_in_targets], V_MIN_LIMIT, V_MAX_LIMIT)
        return pred_unscaled.reshape(-1, len(target_columns))

    true_flat = target_scaler.inverse_transform(pd.DataFrame(y_test.reshape(-1, len(target_columns)), columns=target_columns))
    pred_flat_corrected = apply_physical_correction(y_pred_scaled, X_test_all_batches, I_test_all_batches)

    # Generate timestamps (kept consistent with the original script)
    timestamps = []
    for seg in test_segs:
        t_arr = seg['数据采集时间'].values
        for i in range(0, len(seg)-MIN_LEN+1, TIME_STEPS_OUT):
            timestamps.extend(t_arr[i+TIME_STEPS_IN:i+MIN_LEN])

    result = pd.DataFrame({"Timestamp": timestamps})
    for i, col in enumerate(target_columns):
        result[f"TrueValue_{col}"] = true_flat[:, i]
        result[f"CorrectedPrediction_{col}"] = pred_flat_corrected[:, i]

    now2 = datetime.now()
    dt = now2 - now1
    csv_path = OUTPUT_DIR / f"predictions_{experiment_name}_transformer_spm-constrained-loss_{TIME_STEPS_IN}_{TIME_STEPS_OUT}.csv"
    result.to_csv(csv_path, index=False, encoding="gbk")
    print(f"[{experiment_name}] Saved corrected prediction results to: {csv_path}")

    # Evaluation metrics
    print(f"\n[{experiment_name}] Evaluation metrics for corrected predictions:")
    for i, col in enumerate(target_columns):
        rmse = np.sqrt(np.mean((true_flat[:, i] - pred_flat_corrected[:, i])**2))
        mre = mean_relative_error(true_flat[:, i], pred_flat_corrected[:, i])
        print(f"[{experiment_name}] Test RMSE {col:13s}: {rmse:.6f}")
        print(f"[{experiment_name}] Test MRE  {col:13s}: {mre:.6f}")

    # Save scalers (with experiment-name suffixes for disambiguation)
    joblib.dump(feature_scaler, OUTPUT_DIR / f"feature_scaler_{experiment_name}_in{TIME_STEPS_IN}_out{TIME_STEPS_OUT}.pkl")
    joblib.dump(target_scaler, OUTPUT_DIR / f"target_scaler_{experiment_name}_in{TIME_STEPS_IN}_out{TIME_STEPS_OUT}.pkl")
    print(f"[{experiment_name}] Normalization scalers saved.")

# ---------------------------- Experiment configuration list ----------------------------
# Three experiments: baseline, PE removed, SPM constraint removed
EXPERIMENTS = [
    ("baseline", True, True),   # PE on, SPM loss on
    ("no_pe", False, True),     # PE off, SPM loss on
    ("no_spm", True, False),    # PE on, SPM loss off (main loss only)
]

# Run all experiments in order
if not GENERALIZATION_TEST_ONLY:
    for name, use_pe, use_physical in EXPERIMENTS:
        train_and_evaluate(name, use_pe=use_pe, use_physical=use_physical)
else:
    print("\n" + "="*20 + " Generalization test mode " + "="*20)
    print("Generalization test mode is not fully implemented. Please add generalization logic when GENERALIZATION_TEST_ONLY is set to True.")

print("All experiments completed.")
now3 = datetime.now()
print("Total elapsed:", now3 - now1)
