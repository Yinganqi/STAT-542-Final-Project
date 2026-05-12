from pathlib import Path

import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1. folder paths
project_root = Path(__file__).resolve().parent
data_root = Path("/Users/jessiaw/Desktop/stat542/Project_SVHN")

(project_root / "artifacts").mkdir(exist_ok=True)
(project_root / "figures").mkdir(exist_ok=True)
(project_root / ".mplconfig").mkdir(exist_ok=True)


# 2. read SVHN files
train_data = loadmat(data_root / "train_32x32.mat")
test_data = loadmat(data_root / "test_32x32.mat")

X_train_full = train_data["X"]
y_train_full = train_data["y"]
X_test = test_data["X"]
y_test = test_data["y"]


# 3. fix labels: in SVHN, 10 means digit 0
y_train_full[y_train_full == 10] = 0
y_test[y_test == 10] = 0


# 4. basic inspection
print("X_train_full shape:", X_train_full.shape)
print("y_train_full shape:", y_train_full.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

train_labels = y_train_full.flatten()
test_labels = y_test.flatten()

unique_train, count_train = np.unique(train_labels, return_counts=True)
print("\ntrain label distribution:")
for label, count in zip(unique_train, count_train):
    print(f"digit {label}: {count}")

unique_test, count_test = np.unique(test_labels, return_counts=True)
print("\ntest label distribution:")
for label, count in zip(unique_test, count_test):
    print(f"digit {label}: {count}")

print("\npixel summary:")
print("train pixel min:", X_train_full.min())
print("train pixel max:", X_train_full.max())
print("train pixel mean:", X_train_full.mean())
print("train pixel std:", X_train_full.std())
print("test pixel min:", X_test.min())
print("test pixel max:", X_test.max())
print("test pixel mean:", X_test.mean())
print("test pixel std:", X_test.std())


# 5. save EDA figures
one_image = X_train_full[:, :, :, 0]
one_label = train_labels[0]
plt.imshow(one_image)
plt.title(f"one train image, digit = {one_label}")
plt.savefig(project_root / "figures" / "svhn_one_image.png")
plt.close()

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
digits = sorted(np.unique(train_labels))
for ax, digit in zip(axes.flat, digits):
    index = np.where(train_labels == digit)[0][0]
    image = X_train_full[:, :, :, index]
    ax.imshow(image)
    ax.set_title(f"digit {digit}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(project_root / "figures" / "svhn_all_digits.png")
plt.close()

plt.bar(unique_train, count_train)
plt.title("SVHN train digit distribution")
plt.xlabel("digit")
plt.ylabel("count")
plt.savefig(project_root / "figures" / "svhn_label_distribution.png")
plt.close()


# 6. reorder image shape from (32, 32, 3, N) to (N, 32, 32, 3)
X_train_full = np.transpose(X_train_full, (3, 0, 1, 2))
X_test = np.transpose(X_test, (3, 0, 1, 2))


# 7. normalize pixels from 0-255 to 0-1
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0


# 8. split original train into train and validation for CNN / ViT
X_train_image, X_val_image, y_train, y_val = train_test_split(
    X_train_full,
    train_labels,
    test_size=0.2,
    random_state=542,
    stratify=train_labels,
)


# 9. flatten images for logistic regression / LDA
X_train = X_train_image.reshape(X_train_image.shape[0], -1)
X_val = X_val_image.reshape(X_val_image.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


# 10. standardize vector features for logistic regression / LDA
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test_flat = scaler.transform(X_test_flat)


# 11. save processed data
np.savez_compressed(
    project_root / "artifacts" / "svhn_processed_data.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test_flat,
    y_test=test_labels,
    X_train_image=X_train_image,
    X_val_image=X_val_image,
    X_test_image=X_test,
)


print("\nprocessed data shape:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test_flat.shape)
print("X_train_image:", X_train_image.shape)
print("X_val_image:", X_val_image.shape)
print("X_test_image:", X_test.shape)
print("saved to artifacts/svhn_processed_data.npz")


print("\nEDA summary for report:")
print(
    "The SVHN dataset contains 73,257 training samples and 26,032 test samples. "
    "Each observation is a 32x32 RGB image of a house-number digit collected from real street-view scenes."
)
print(
    "The classification task has 10 classes, where the original label 10 is remapped to digit 0. "
    f"Class counts in the training split range from {count_train.min()} to {count_train.max()}, "
    "so the dataset is moderately imbalanced."
)
print(
    f"Pixel intensities range from {X_train_full.min():.1f} to {X_train_full.max():.1f} after normalization, "
    "and the images contain natural background variation, color information, and real-world noise."
)
print(
    "For preprocessing, we normalize pixel values to [0, 1], create a stratified 80/20 train-validation split with random_state=542, "
    "flatten images for linear models such as logistic regression and LDA, and retain image tensors for CNN- and ViT-based models."
)
