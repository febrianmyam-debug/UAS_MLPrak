"""
=========================================================
HeartWise MLP Regression Model
---------------------------------------------------------
Multilayer Perceptron (MLP) for Heart Disease Risk Score
Prediction.
=========================================================
"""

import joblib
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from src.config import (
    MLP_MODEL_FILE,
    MLP_CONFIG,
    MLP_HISTORY_FILE
)

from src.callbacks import get_callbacks
from src.utils import log, set_seed

# =========================================================
# BUILD MODEL
# =========================================================

def build_model(input_dim):
    """
    Build MLP model using Functional API.
    """

    set_seed()

    inputs = Input(
        shape=(input_dim,),
        name="Input"
    )

    x = Dense(
        128,
        activation="relu"
    )(inputs)

    x = BatchNormalization()(x)

    x = Dropout(0.30)(x)

    x = Dense(
        64,
        activation="relu"
    )(x)

    x = BatchNormalization()(x)

    x = Dropout(0.30)(x)

    x = Dense(
        32,
        activation="relu"
    )(x)

    outputs = Dense(
        1,
        activation="linear",
        name="RiskScore"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="HeartWiseMLP"
    )

    log("MLP model created.")

    return model

# =========================================================
# COMPILE MODEL
# =========================================================

def compile_model(model):
    """
    Compile MLP model.
    """

    optimizer = Adam(
        learning_rate=MLP_CONFIG["learning_rate"]
    )

    model.compile(

        optimizer=optimizer,

        loss="mse",

        metrics=[

            "mae",

            tf.keras.metrics.RootMeanSquaredError(
                name="rmse"
            )

        ]

    )

    log("MLP model compiled.")

    return model

# =========================================================
# MODEL SUMMARY
# =========================================================

def show_summary(model):
    """
    Display model summary.
    """

    model.summary()

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
    """
    Train the MLP model.
    """

    log("Starting MLP training...")

    callbacks = get_callbacks()

    history = model.fit(

        X_train,

        y_train,

        validation_data=(

            X_valid,

            y_valid

        ),

        epochs=MLP_CONFIG["epochs"],

        batch_size=MLP_CONFIG["batch_size"],

        callbacks=callbacks,

        verbose=1

    )

    log("Training completed.")

    return history

# =========================================================
# SAVE MODEL
# =========================================================

def save_model(model):
    """
    Save trained model.
    """

    model.save(MLP_MODEL_FILE)

    log("MLP model saved.")

# =========================================================
# SAVE HISTORY
# =========================================================

def save_history(history, filepath):
    """
    Save training history.
    """

    joblib.dump(
        history.history,
        filepath
    )

    log("Training history saved.")


# =========================================================
# LOAD SAVED MODEL
# =========================================================

def load_saved_model():
    """
    Load trained MLP model.
    """

    log("Loading trained MLP model...")

    model = load_model(
        MLP_MODEL_FILE
    )

    log("Model loaded successfully.")

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
        X,
        verbose=0
    )

    return prediction.flatten()


# =========================================================
# EVALUATE
# =========================================================

def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate trained model.
    """

    result = model.evaluate(

        X_test,

        y_test,

        verbose=0

    )

    metrics = {

        "Loss": result[0],

        "MAE": result[1],

        "RMSE": result[2]

    }

    log("Evaluation completed.")

    return metrics

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
    Complete MLP training pipeline.
    """

    log("=" * 60)
    log("START MLP TRAINING")
    log("=" * 60)

    model = build_model(
        X_train.shape[1]
    )

    compile_model(model)

    show_summary(model)

    history = train_model(

        model,

        X_train,

        y_train,

        X_valid,

        y_valid

    )

    save_model(model)

    save_history(
        history,
        MLP_HISTORY_FILE
    )

    log("=" * 60)
    log("MLP TRAINING FINISHED")
    log("=" * 60)

    return model, history

