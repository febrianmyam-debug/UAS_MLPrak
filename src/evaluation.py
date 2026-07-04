"""
=========================================================
HeartWise Model Evaluation
---------------------------------------------------------
Evaluation metrics for regression models.
=========================================================
"""

import pandas as pd
import numpy as np

from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    mean_absolute_percentage_error,

    r2_score

)

from src.utils import log

# =========================================================
# MAE
# =========================================================

def mae(
    y_true,
    y_pred
):
    """
    Mean Absolute Error.
    """

    return mean_absolute_error(
        y_true,
        y_pred
    )

# =========================================================
# RMSE
# =========================================================

def rmse(
    y_true,
    y_pred
):
    """
    Root Mean Squared Error.
    """

    return np.sqrt(

        mean_squared_error(

            y_true,

            y_pred

        )

    )

# =========================================================
# MAPE
# =========================================================

def mape(
    y_true,
    y_pred
):
    """
    Mean Absolute Percentage Error.
    """

    return mean_absolute_percentage_error(

        y_true,

        y_pred

    ) * 100

# =========================================================
# R2 SCORE
# =========================================================

def r2(
    y_true,
    y_pred
):
    """
    R2 Score.
    """

    return r2_score(

        y_true,

        y_pred

    )

# =========================================================
# REGRESSION METRICS
# =========================================================

def regression_metrics(
    y_true,
    y_pred
):
    """
    Calculate all regression metrics.
    """

    results = {

        "MAE": mae(
            y_true,
            y_pred
        ),

        "RMSE": rmse(
            y_true,
            y_pred
        ),

        "MAPE": mape(
            y_true,
            y_pred
        ),

        "R2": r2(
            y_true,
            y_pred
        )

    }

    log("Evaluation completed.")

    return pd.DataFrame(
        results,
        index=["MLP"]
    )

# =========================================================
# EVALUATE REGRESSION
# =========================================================

def evaluate_regression(
    y_true,
    y_pred
):
    """
    Evaluate regression model.
    """

    result = {

        "MAE": mae(

            y_true,

            y_pred

        ),

        "RMSE": rmse(

            y_true,

            y_pred

        ),

        "MAPE": mape(

            y_true,

            y_pred

        ),

        "R2": r2(

            y_true,

            y_pred

        )

    }

    return result

# =========================================================
# SAVE RESULT
# =========================================================

def save_result(
    comparison,
    filepath
):
    """
    Save comparison result to CSV.
    """

    comparison.to_csv(

        filepath,

        index=False

    )

    log("Evaluation result saved.")

# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(
    comparison
):
    """
    Display comparison result.
    """

    print()

    print("=" * 60)

    print("MODEL COMPARISON")

    print("=" * 60)

    print(comparison)

    print("=" * 60)

# =========================================================
# BEST MODEL
# =========================================================

def best_model(
    comparison
):
    """
    Determine the best model.
    """

    mlp_score = 0

    tabnet_score = 0

    # Lower is better
    for metric in ["MAE", "RMSE", "MAPE"]:

        value = comparison.loc[
            comparison["Metric"] == metric
        ]

        mlp = value["MLP"].values[0]

        tabnet = value["TabNet"].values[0]

        if mlp < tabnet:

            mlp_score += 1

        else:

            tabnet_score += 1

    # Higher is better
    value = comparison.loc[
        comparison["Metric"] == "R2"
    ]

    mlp = value["MLP"].values[0]

    tabnet = value["TabNet"].values[0]

    if mlp > tabnet:

        mlp_score += 1

    else:

        tabnet_score += 1

    print()

    print("=" * 60)

    print("BEST MODEL")

    print("=" * 60)

    print(f"MLP Score     : {mlp_score}")

    print(f"TabNet Score  : {tabnet_score}")

    print()

    if mlp_score > tabnet_score:

        print("Best Model : MLP")

    elif tabnet_score > mlp_score:

        print("Best Model : TabNet")

    else:

        print("Both models perform equally.")

    print("=" * 60)

# =========================================================
# EVALUATION PIPELINE
# =========================================================

def evaluation_pipeline(
    y_true,
    mlp_prediction,
    tabnet_prediction,
    save_path=None
):
    """
    Complete evaluation pipeline.
    """

    log("=" * 60)
    log("START MODEL EVALUATION")
    log("=" * 60)

    mlp_result = evaluate_regression(

        y_true,

        mlp_prediction

    )

    tabnet_result = evaluate_regression(

        y_true,

        tabnet_prediction

    )

    comparison = compare_models(

        mlp_result,

        tabnet_result

    )

    print_summary(
        comparison
    )

    best_model(
        comparison
    )

    if save_path is not None:

        save_result(

            comparison,

            save_path

        )

    log("=" * 60)
    log("MODEL EVALUATION FINISHED")
    log("=" * 60)

    return comparison

# =========================================================
# COMPARE MODEL
# =========================================================

def compare_models(mlp_result, tabnet_result):

    comparison = pd.DataFrame({

        "Metric": ["MAE", "RMSE", "MAPE", "R2"],

        "MLP": [
            mlp_result["MAE"],
            mlp_result["RMSE"],
            mlp_result["MAPE"],
            mlp_result["R2"]
        ],

        "TabNet": [
            tabnet_result["MAE"],
            tabnet_result["RMSE"],
            tabnet_result["MAPE"],
            tabnet_result["R2"]
        ]

    })

    return comparison