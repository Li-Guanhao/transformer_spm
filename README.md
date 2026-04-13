# Battery-Risk-Transformer

A Transformer-based battery sequence forecasting project with an optional SPM-like physical constraint in the loss function.

## Repository Structure

```
Battery-Risk-Transformer/
├── data_preprocessing.py
├── model/
│   ├── __init__.py
│   ├── transformer.py
│   └── spm_constraint.py
├── train.py
├── evaluate.py
├── config.yaml
└── README.md
```

## Features

- Transformer forecaster for multivariate time-series prediction
- Optional positional encoding ablation
- Optional SPM-like physics constraint ablation
- Sliding-window sample generation with segment splitting
- Train/validation/test split by segment
- Prediction export and scaler persistence

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- pandas
- scikit-learn
- tqdm
- PyYAML
- joblib

You can install them with the provided `requirements.txt` file.

## Data Format

The input CSV must contain the columns defined in `config.yaml`. The exact column names are kept in the configuration file because they follow the original dataset schema.

Rows are split into segments when the time gap between adjacent samples is greater than `max_gap_seconds`.

## Configuration

All paths, model hyperparameters, training settings, and physics parameters are stored in `config.yaml`.

Update the following fields before running:

- `paths.data_file`
- `paths.external_test_file` if needed
- `paths.output_dir`

## Training

Run:

```bash
python train.py
```

This will:

1. Load and segment the dataset
2. Fit scalers on the training data
3. Train the configured experiments
4. Save checkpoints, predictions, and scalers

## Evaluation

Run:

```bash
python evaluate.py
```

This will load saved checkpoints and scaler files, then export evaluation CSV files for each experiment.

## Outputs

Results are written to the directories defined in `config.yaml`:

- `outputs/models/` for checkpoints
- `outputs/predictions/` for prediction CSV files
- `outputs/scalers/` for saved scalers

## Notes

- The generalization-only mode from the original script is not implemented here.
