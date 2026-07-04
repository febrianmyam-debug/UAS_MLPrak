"""
=========================================================
HeartWise Callbacks
---------------------------------------------------------
TensorFlow/Keras callbacks used during model training.
=========================================================
"""

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from src.config import (
    MLP_MODEL_FILE,
    MLP_CONFIG
)


# =========================================================
# EARLY STOPPING
# =========================================================

def early_stopping():
    """
    Stop training when validation loss stops improving.
    """

    return EarlyStopping(

        monitor="val_loss",

        patience=MLP_CONFIG["patience"],

        restore_best_weights=True,

        verbose=1

    )


# =========================================================
# MODEL CHECKPOINT
# =========================================================

def model_checkpoint():
    """
    Save the best model during training.
    """

    return ModelCheckpoint(

        filepath=MLP_MODEL_FILE,

        monitor="val_loss",

        save_best_only=True,

        save_weights_only=False,

        verbose=1

    )


# =========================================================
# REDUCE LEARNING RATE
# =========================================================

def reduce_lr():
    """
    Reduce learning rate when validation loss plateaus.
    """

    return ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=5,

        min_lr=1e-6,

        verbose=1

    )


# =========================================================
# GET CALLBACKS
# =========================================================

def get_callbacks():
    """
    Return all callbacks used during training.
    """

    callbacks = [

        early_stopping(),

        model_checkpoint(),

        reduce_lr()

    ]

    return callbacks