import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Set path
project_root = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))

# Load preprocessed data
data = np.load(project_root / "artifacts" / "svhn_processed_data.npz")

# FCN data: Flattened format (3072 features)
X_train_flat = data['X_train']  # Shape: (58605, 3072)
y_train = data['y_train']
X_val_flat = data['X_val']      # Shape: (14651, 3072)
y_val = data['y_val']
X_test_flat = data['X_test']    # Shape: (13068, 3072)
y_test = data['y_test']

print("FCN data shape:")
print(f"X_train_flat: {X_train_flat.shape}")
print(f"y_train: {y_train.shape}")

# Convert labels to categorical format
num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# ========================================
# FCN (Fully Connected Neural Network) Model Definition
# ========================================
def create_fully_connected_model(input_shape):
    """Create fully connected neural network

    FCN Architecture:
    - Input layer: 3072 neurons (32x32x3 flattened)
    - Hidden layer 1: 512 neurons + BatchNorm + ReLU + Dropout(0.3)
    - Hidden layer 2: 256 neurons + BatchNorm + ReLU + Dropout(0.3)
    - Hidden layer 3: 128 neurons + BatchNorm + ReLU + Dropout(0.3)
    - Output layer: 10 neurons (Softmax activation)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Optimizer with weight decay (L2 regularization)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ========================================
# Training and Evaluation Function
# ========================================
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train model and return results dictionary"""

    # Learning rate scheduler callback
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # Record training start time
    start_time = time.time()

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=256,
        callbacks=[lr_scheduler],
        verbose=1
    )

    training_time = time.time() - start_time

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Calculate inference time (on 100 samples)
    inference_start = time.time()
    _ = model.predict(X_test[:100], verbose=0)
    inference_time_per_sample = (time.time() - inference_start) / 100 * 1000  # milliseconds

    # Get model parameter count
    total_params = model.count_params()

    # Return results
    return {
        'history': history.history,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'training_time': training_time,
        'inference_time_per_sample': inference_time_per_sample,
        'total_params': total_params,
        'model': model
    }

# ========================================
# Train FCN Model
# ========================================
print("\n=== Start Training FCN Model ===")

# Create FCN model
fcnn_model = create_fully_connected_model((X_train_flat.shape[1],))

# Training and evaluation
fcnn_results = train_and_evaluate_model(
    fcnn_model,
    X_train_flat, y_train_cat,
    X_val_flat, y_val_cat,
    X_test_flat, y_test_cat
)

print(f"Test Accuracy: {fcnn_results['test_accuracy']:.4f}")
print(f"Training time: {fcnn_results['training_time']:.2f}s")
print(f"Inference time: {fcnn_results['inference_time_per_sample']:.2f}ms/sample")
print(f"Total parameters: {fcnn_results['total_params']:,}")

# ========================================
# Save FCN Model Results to File
# ========================================
results_file = project_root / "FCN_model_results.txt"

with open(results_file, 'w', encoding='utf-8') as f:
    f.write("FCN (Fully Connected Neural Network) Model Results\n")
    f.write("=" * 50 + "\n\n")

    f.write("Model Architecture:\n")
    f.write("- Input layer: 3072 neurons (32x32x3 image flattened)\n")
    f.write("- Hidden layer 1: 512 neurons + BatchNorm + ReLU + Dropout(0.3)\n")
    f.write("- Hidden layer 2: 256 neurons + BatchNorm + ReLU + Dropout(0.3)\n")
    f.write("- Hidden layer 3: 128 neurons + BatchNorm + ReLU + Dropout(0.3)\n")
    f.write("- Output layer: 10 neurons + Softmax\n\n")

    f.write("Training Configuration:\n")
    f.write("- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)\n")
    f.write("- Loss function: Categorical Crossentropy\n")
    f.write("- Batch size: 256\n")
    f.write("- Max training epochs: 15\n")
    f.write("- Early stopping: None\n")
    f.write("- Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=2)\n\n")

    f.write("Performance Metrics:\n")
    f.write(f"- Training time: {fcnn_results['training_time']:.2f}s\n")
    f.write(f"- Inference time: {fcnn_results['inference_time_per_sample']:.2f}ms/sample\n")
    f.write(f"- Total parameters: {fcnn_results['total_params']:,}\n\n")

    # Training history
    f.write("Training History (Last 5 epochs):\n")
    epochs = len(fcnn_results['history']['accuracy'])
    for i in range(max(0, epochs-5), epochs):
        f.write(f"Epoch {i+1}: train acc={fcnn_results['history']['accuracy'][i]:.4f}, "
                f"val acc={fcnn_results['history']['val_accuracy'][i]:.4f}, "
                f"train loss={fcnn_results['history']['loss'][i]:.4f}, "
                f"val loss={fcnn_results['history']['val_loss'][i]:.4f}\n")

    # Summary table
    f.write("\nSummary table\n")
    f.write("epoch | train_acc | val_acc | train_loss | val_loss\n")
    for i in range(epochs):
        f.write(f"{i+1:5d} | {fcnn_results['history']['accuracy'][i]:.4f} | {fcnn_results['history']['val_accuracy'][i]:.4f} | {fcnn_results['history']['loss'][i]:.4f} | {fcnn_results['history']['val_loss'][i]:.4f}\n")

print(f"\n✓ FCN model results saved to: {results_file}")

# ========================================
# Optional: Save FCN Model Weights
# ========================================
model_save_path = project_root / "artifacts" / "fcn_model.h5"
fcnn_model.save(model_save_path)
print(f"✓ FCN model saved to: {model_save_path}")
