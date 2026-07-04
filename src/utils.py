"""
=========================================================
HeartWise Utility Functions
---------------------------------------------------------
Reusable utility functions for the HeartWise project.
=========================================================
"""

import random
import joblib
import numpy as np
import tensorflow as tf

from pathlib import Path
from datetime import datetime

from src.config import RANDOM_STATE


# =========================================================
# RANDOM SEED
# =========================================================

def set_seed(seed: int = RANDOM_STATE):
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================================================
# DIRECTORY
# =========================================================

def create_directory(directory):
    """
    Create directory if it does not exist.

    Parameters
    ----------
    directory : Path
    """

    Path(directory).mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# SAVE PICKLE
# =========================================================

def save_pickle(obj, filepath):
    """
    Save object into pickle file.
    """

    joblib.dump(obj, filepath)


# =========================================================
# LOAD PICKLE
# =========================================================

def load_pickle(filepath):
    """
    Load pickle object.
    """

    return joblib.load(filepath)


# =========================================================
# PRINT SECTION
# =========================================================

def print_section(title):
    """
    Print section title.
    """

    print("=" * 60)
    print(title)
    print("=" * 60)


# =========================================================
# CURRENT TIME
# =========================================================

def current_time():
    """
    Return current timestamp.
    """

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# =========================================================
# LOG MESSAGE
# =========================================================

def log(message):
    """
    Print log message with timestamp.
    """

    print(f"[{current_time()}] {message}")


# =========================================================
# CHECK FILE
# =========================================================

def file_exists(filepath):
    """
    Check whether file exists.
    """

    return Path(filepath).exists()


# =========================================================
# CHECK DIRECTORY
# =========================================================

def directory_exists(directory):
    """
    Check whether directory exists.
    """

    return Path(directory).exists()