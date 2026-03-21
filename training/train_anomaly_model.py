"""Train a vibration anomaly detection model for embedded deployment.

Exports a TensorFlow Lite INT8 quantized model for STM32 inference.
"""
import numpy as np
import tensorflow as tf


def generate_synthetic_data(n_normal=2000, n_anomaly=400, seq_len=64):
    """Synthetic accelerometer data (normal + anomaly vibration)."""
    X, y = [], []
    for _ in range(n_normal):
        sig = np.random.normal(0, 0.3, seq_len)          # normal vibration
        X.append(sig); y.append(0)
    for _ in range(n_anomaly):
        sig = np.random.normal(0, 1.5, seq_len)          # fault vibration
        sig[np.random.randint(20, 44)] += np.random.uniform(3, 6)
        X.append(sig); y.append(1)
    X = np.array(X, dtype=np.float32)[..., np.newaxis]   # (N, 64, 1)
    y = np.array(y, dtype=np.float32)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def build_model(seq_len=64):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, 1)),
        tf.keras.layers.Conv1D(16, 5, activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])


def export_tflite(model, X_train):
    """Quantize to INT8 for MCU deployment."""
    def rep_gen():
        for x in X_train[:200]:
            yield [x[np.newaxis]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open("anomaly_model_int8.tflite", "wb") as f:
        f.write(tflite_model)
    print(f"✅ Exported INT8 model: {len(tflite_model)/1024:.1f} KB")


if __name__ == "__main__":
    X, y = generate_synthetic_data()
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_data=(X_test, y_test))

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc*100:.1f}%")
    export_tflite(model, X_train)
