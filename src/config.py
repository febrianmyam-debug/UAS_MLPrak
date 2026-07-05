"""
=========================================================
HeartWise Configuration
---------------------------------------------------------
Project :
HeartWise - Heart Disease Risk Score Prediction

Author :
Kelompok ...

Description :
Configuration file for all project settings.
=========================================================
"""

from pathlib import Path

# =========================================================
# PROJECT DIRECTORY
# =========================================================

# Root Folder
ROOT_DIR = Path(__file__).resolve().parent.parent

# =========================================================
# DATASET
# =========================================================

DATASET_DIR = ROOT_DIR / "dataset"

DATASET_FILE = DATASET_DIR / "heartv1.csv"

TRAIN_SCALED_FILE = DATASET_DIR / "train_scaled.csv"
TEST_SCALED_FILE = DATASET_DIR / "test_scaled.csv"

TRAIN_RAW_FILE = DATASET_DIR / "train_raw.csv"
VALID_RAW_FILE = DATASET_DIR / "valid_raw.csv"
TEST_RAW_FILE = DATASET_DIR / "test_raw.csv"

TRAIN_SCALED_FILE = DATASET_DIR / "train_scaled.csv"
VALID_SCALED_FILE = DATASET_DIR / "valid_scaled.csv"
TEST_SCALED_FILE = DATASET_DIR / "test_scaled.csv"

# =========================================================
# MODEL DIRECTORY
# =========================================================

MODEL_DIR = ROOT_DIR / "models"

MLP_MODEL_FILE = MODEL_DIR / "mlp.keras"

TABNET_MODEL_FILE = MODEL_DIR / "tabnet_model"

SCALER_FILE = MODEL_DIR / "scaler.pkl"

FEATURE_FILE = MODEL_DIR / "feature_columns.pkl"

MLP_HISTORY_FILE = MODEL_DIR / "mlp_history.pkl"

TABNET_HISTORY_FILE = MODEL_DIR / "tabnet_history.pkl"

# =========================================================
# TARGET COLUMN
# =========================================================

TARGET_COLUMN = "Heart Disease Risk Score"

# Kolom klasifikasi (tidak digunakan)
DROP_COLUMNS = [
    "target"
]

# =========================================================
# RANDOM SEED
# =========================================================

RANDOM_STATE = 42

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

TRAIN_SIZE = 0.70
VALID_SIZE = 0.15
TEST_SIZE = 0.15

# =========================================================
# SCALING
# =========================================================

USE_STANDARD_SCALER = True

# =========================================================
# MLP CONFIGURATION
# =========================================================

MLP_CONFIG = {

    "epochs": 150,

    "batch_size": 32,

    "learning_rate": 0.001,

    "patience": 20

}

# =========================================================
# TABNET CONFIGURATION
# =========================================================

TABNET_CONFIG = {
    "n_d": 16,
    "n_a": 16,
    "n_steps": 3,
    "gamma": 1.3,
    "lambda_sparse": 1e-4,
    "optimizer_lr": 0.001,
    "max_epochs": 200,
    "patience": 30,
    "batch_size": 32,
    "virtual_batch_size": 16
}
# =========================================================
# EVALUATION METRICS
# =========================================================

EVALUATION_METRICS = [

    "MAE",

    "RMSE",

    "MAPE",

    "R2"

]

# =========================================================
# FIGURE CONFIGURATION
# =========================================================

FIGURE_SIZE = (10,6)

FIGURE_DPI = 120

# =========================================================
# SAVE FIGURE
# =========================================================

SAVE_FIGURE = True

RESULT_DIR = ROOT_DIR / "results"

DATASET_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# API
# =========================================================

HOST = "127.0.0.1"

PORT = 5000

DEBUG = True

PROJECT_NAME = "HeartWise"

VERSION = "1.0.0"