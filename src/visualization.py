"""
=========================================================
HeartWise Visualization
---------------------------------------------------------
Visualization functions for EDA and Model Evaluation.
=========================================================
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    FIGURE_SIZE,
    FIGURE_DPI
)

# =========================================================
# DISTRIBUTION
# =========================================================

def plot_distribution(df):

    """
    Plot histogram for every numerical feature.
    """

    df.hist(

        figsize=(18,12),

        bins=20

    )

    plt.tight_layout()

    plt.show()

# =========================================================
# CORRELATION
# =========================================================

def plot_correlation(df):

    """
    Plot correlation heatmap.
    """

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(12,10))

    plt.imshow(corr)

    plt.xticks(

        range(len(corr.columns)),

        corr.columns,

        rotation=90

    )

    plt.yticks(

        range(len(corr.columns)),

        corr.columns

    )

    plt.colorbar()

    plt.tight_layout()

    plt.show()

# =========================================================
# BOXPLOT
# =========================================================

def plot_boxplot(df):

    """
    Plot boxplot for numerical features.
    """

    df.plot(

        kind="box",

        figsize=(18,8)

    )

    plt.tight_layout()

    plt.show()

# =========================================================
# LOSS CURVE
# =========================================================

def plot_loss(history):

    """
    Plot training history.
    """

    plt.figure(figsize=FIGURE_SIZE)

    plt.plot(

        history.history["loss"],

        label="Train"

    )

    plt.plot(

        history.history["val_loss"],

        label="Validation"

    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.legend()

    plt.grid(True)

    plt.show()

# =========================================================
# PREDICTION
# =========================================================

def plot_prediction(

    y_true,

    y_pred

):

    """
    Actual vs Prediction.
    """

    plt.figure(figsize=FIGURE_SIZE)

    plt.scatter(

        y_true,

        y_pred,

        alpha=0.7

    )

    plt.xlabel("Actual")

    plt.ylabel("Prediction")

    plt.title("Actual vs Prediction")

    plt.grid(True)

    plt.show()

# =========================================================
# RESIDUAL
# =========================================================

def plot_residual(

    y_true,

    y_pred

):

    """
    Plot residual.
    """

    residual = y_true-y_pred

    plt.figure(figsize=FIGURE_SIZE)

    plt.scatter(

        y_pred,

        residual

    )

    plt.axhline(

        0,

        linestyle="--"

    )

    plt.xlabel("Prediction")

    plt.ylabel("Residual")

    plt.title("Residual Plot")

    plt.grid(True)

    plt.show()

# =========================================================
# MODEL COMPARISON
# =========================================================

def plot_comparison(comparison):

    """
    Plot MLP vs TabNet.
    """

    comparison.set_index(

        "Metric"

    ).plot(

        kind="bar",

        figsize=FIGURE_SIZE

    )

    plt.ylabel("Score")

    plt.grid(True)

    plt.tight_layout()

    plt.show()

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def plot_feature_importance(

    importance,

    feature_names

):

    """
    Plot TabNet feature importance.
    """

    importance = pd.Series(

        importance,

        index=feature_names

    )

    importance.sort_values().plot(

        kind="barh",

        figsize=(10,8)

    )

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.show()

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def plot_feature_importance(

    importance,

    feature_names

):

    """
    Plot TabNet feature importance.
    """

    importance = pd.Series(

        importance,

        index=feature_names

    )

    importance.sort_values().plot(

        kind="barh",

        figsize=(10,8)

    )

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.show()

