"""
=========================================================
HeartWise TabNet Regression Model
---------------------------------------------------------
TabNet Regressor for Heart Disease Risk Score Prediction
=========================================================
"""

import joblib
import numpy as np

from pytorch_tabnet.tab_model import TabNetRegressor

from src.config import (
    TABNET_CONFIG,
    TABNET_MODEL_FILE,
    TABNET_HISTORY_FILE
)

from src.utils import (
    log,
    set_seed
)

# =========================================================
# BUILD MODEL
# =========================================================

def build_model():
    """
    Build TabNet Regressor.
    """

    set_seed()

    model = TabNetRegressor(

        n_d=TABNET_CONFIG["n_d"],

        n_a=TABNET_CONFIG["n_a"],

        n_steps=TABNET_CONFIG["n_steps"],

        gamma=TABNET_CONFIG["gamma"],

        lambda_sparse=TABNET_CONFIG["lambda_sparse"],

        optimizer_params=dict(

            lr=TABNET_CONFIG["optimizer_lr"]

        ),

        verbose=1

    )

    log("TabNet model created.")

    return model

# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid
):

    log("Starting TabNet training...")

    model.fit(
        X_train=np.asarray(X_train, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.float32).reshape(-1,1),

        eval_set=[
            (
                np.asarray(X_valid, dtype=np.float32),
                np.asarray(y_valid, dtype=np.float32).reshape(-1,1)
            )
        ],

        max_epochs=TABNET_CONFIG["max_epochs"],
        patience=TABNET_CONFIG["patience"],
        batch_size=TABNET_CONFIG["batch_size"],
        virtual_batch_size=TABNET_CONFIG["virtual_batch_size"]
    )

    log("Training completed.")

    history = model.history

    return model, history
# =========================================================
# SAVE MODEL
# =========================================================

def save_model(model):
    """
    Save TabNet model.
    """

    model.save_model(
        TABNET_MODEL_FILE
    )

    log("TabNet model saved.")

# =========================================================
# LOAD MODEL
# =========================================================

def load_saved_model():
    """
    Load trained TabNet model.
    """

    model = TabNetRegressor()

    model.load_model(
        f"{TABNET_MODEL_FILE}.zip"
    )

    log("TabNet model loaded.")

    return model

# =========================================================
# PREDICT
# =========================================================

def predict(
    model,
    X
):
    """
    Predict Heart Disease Risk Score.
    """

    prediction = model.predict(

        np.asarray(
            X,
            dtype=np.float32
        )

    )

    return prediction.flatten()

# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate TabNet model.
    """

    prediction = predict(
        model,
        X_test
    )

    result = {

        "Prediction": prediction,

        "Ground Truth": y_test

    }

    log("Evaluation completed.")

    return result

# =========================================================
# SAVE HISTORY
# =========================================================

def save_history(
    model,
    filepath=TABNET_HISTORY_FILE
):
    """
    Save training history.
    """

    joblib.dump(
        model.history,
        filepath
    )

    log("Training history saved.")

# =========================================================
# TRAINING PIPELINE
# =========================================================

def train_pipeline(
    X_train,
    y_train,
    X_valid,
    y_valid
):
    """
    Complete TabNet training pipeline.
    """

    log("=" * 60)
    log("START TABNET TRAINING")
    log("=" * 60)

    model = build_model()

    model, history = train_model(

        model,

        X_train,

        y_train,

        X_valid,

        y_valid

    )

    save_model(model)

    save_history(model)

    log("=" * 60)
    log("TABNET TRAINING FINISHED")
    log("=" * 60)

    return model, history

