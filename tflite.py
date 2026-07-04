import tensorflow as tf

from src.config import (
    MLP_MODEL_FILE,
    MODEL_DIR
)
# ===========================================
# LOAD MODEL
# ===========================================

model = tf.keras.models.load_model(
    MLP_MODEL_FILE
)

print("Model loaded.")

# ===========================================
# CONVERT
# ===========================================

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

tflite_model = converter.convert()

# ===========================================
# SAVE
# ===========================================

output_path = MODEL_DIR / "heartwise_mlp.tflite"

with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at: {output_path}")

print("Model loaded.")

# ===========================================
# CONVERT
# ===========================================

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

tflite_model = converter.convert()

# ===========================================
# SAVE
# ===========================================

output_path = MODEL_DIR / "heartwise_mlp.tflite"

with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at: {output_path}")