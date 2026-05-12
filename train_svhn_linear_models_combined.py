from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


# 1. read processed SVHN data
project_root = Path(__file__).resolve().parent
data = np.load(project_root / "artifacts" / "svhn_processed_data.npz")

X_train_full = data["X_train"]
y_train_full = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]


# 2. subsample baseline without PCA
X_train_sub, _, y_train_sub, _ = train_test_split(
    X_train_full,
    y_train_full,
    train_size=15000,
    random_state=542,
    stratify=y_train_full,
)

print("subsample experiment")
print("full training size:", X_train_full.shape[0])
print("subsampled training size:", X_train_sub.shape[0])
print("number of original features:", X_train_sub.shape[1])

logreg_sub = LogisticRegression(max_iter=1000, random_state=542)
logreg_sub.fit(X_train_sub, y_train_sub)

train_pred_logreg_sub = logreg_sub.predict(X_train_sub)
val_pred_logreg_sub = logreg_sub.predict(X_val)
test_pred_logreg_sub = logreg_sub.predict(X_test)

logreg_sub_train_acc = accuracy_score(y_train_sub, train_pred_logreg_sub)
logreg_sub_val_acc = accuracy_score(y_val, val_pred_logreg_sub)
logreg_sub_test_acc = accuracy_score(y_test, test_pred_logreg_sub)
logreg_sub_test_f1 = f1_score(y_test, test_pred_logreg_sub, average="macro")
logreg_sub_gap = logreg_sub_train_acc - logreg_sub_val_acc
logreg_sub_cm = confusion_matrix(y_test, test_pred_logreg_sub)

lda_sub = LinearDiscriminantAnalysis()
lda_sub.fit(X_train_sub, y_train_sub)

train_pred_lda_sub = lda_sub.predict(X_train_sub)
val_pred_lda_sub = lda_sub.predict(X_val)
test_pred_lda_sub = lda_sub.predict(X_test)

lda_sub_train_acc = accuracy_score(y_train_sub, train_pred_lda_sub)
lda_sub_val_acc = accuracy_score(y_val, val_pred_lda_sub)
lda_sub_test_acc = accuracy_score(y_test, test_pred_lda_sub)
lda_sub_test_f1 = f1_score(y_test, test_pred_lda_sub, average="macro")
lda_sub_gap = lda_sub_train_acc - lda_sub_val_acc
lda_sub_cm = confusion_matrix(y_test, test_pred_lda_sub)

print("\nLogistic Regression with subsampled training data")
print("train accuracy:", round(logreg_sub_train_acc, 4))
print("validation accuracy:", round(logreg_sub_val_acc, 4))
print("test accuracy:", round(logreg_sub_test_acc, 4))
print("test macro F1:", round(logreg_sub_test_f1, 4))
print("generalization gap:", round(logreg_sub_gap, 4))
print("confusion matrix:")
print(logreg_sub_cm)

print("\nLDA with subsampled training data")
print("train accuracy:", round(lda_sub_train_acc, 4))
print("validation accuracy:", round(lda_sub_val_acc, 4))
print("test accuracy:", round(lda_sub_test_acc, 4))
print("test macro F1:", round(lda_sub_test_f1, 4))
print("generalization gap:", round(lda_sub_gap, 4))
print("confusion matrix:")
print(lda_sub_cm)


# 3. PCA experiments on full train/validation/test
pca_dims = [20, 50, 100, 200, 300]
results = []
logreg_cm_dim100 = None
lda_cm_dim100 = None

print("\nPCA comparison")

for n_components in pca_dims:
    pca = PCA(n_components=n_components, random_state=542)
    X_train_pca = pca.fit_transform(X_train_full)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    explained = pca.explained_variance_ratio_.sum()

    logreg = LogisticRegression(max_iter=1000, random_state=542)
    logreg.fit(X_train_pca, y_train_full)

    train_pred_logreg = logreg.predict(X_train_pca)
    val_pred_logreg = logreg.predict(X_val_pca)
    test_pred_logreg = logreg.predict(X_test_pca)

    logreg_train_acc = accuracy_score(y_train_full, train_pred_logreg)
    logreg_val_acc = accuracy_score(y_val, val_pred_logreg)
    logreg_test_acc = accuracy_score(y_test, test_pred_logreg)
    logreg_test_f1 = f1_score(y_test, test_pred_logreg, average="macro")
    logreg_gap = logreg_train_acc - logreg_val_acc
    logreg_cm = confusion_matrix(y_test, test_pred_logreg)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_pca, y_train_full)

    train_pred_lda = lda.predict(X_train_pca)
    val_pred_lda = lda.predict(X_val_pca)
    test_pred_lda = lda.predict(X_test_pca)

    lda_train_acc = accuracy_score(y_train_full, train_pred_lda)
    lda_val_acc = accuracy_score(y_val, val_pred_lda)
    lda_test_acc = accuracy_score(y_test, test_pred_lda)
    lda_test_f1 = f1_score(y_test, test_pred_lda, average="macro")
    lda_gap = lda_train_acc - lda_val_acc
    lda_cm = confusion_matrix(y_test, test_pred_lda)

    if n_components == 100:
        logreg_cm_dim100 = logreg_cm
        lda_cm_dim100 = lda_cm

    results.append(
        {
            "pca_dim": n_components,
            "explained_variance": explained,
            "logreg_val_acc": logreg_val_acc,
            "logreg_test_acc": logreg_test_acc,
            "logreg_test_f1": logreg_test_f1,
            "logreg_gap": logreg_gap,
            "lda_val_acc": lda_val_acc,
            "lda_test_acc": lda_test_acc,
            "lda_test_f1": lda_test_f1,
            "lda_gap": lda_gap,
        }
    )

    print(f"\nPCA dimension = {n_components}")
    print("explained variance ratio:", round(explained, 4))
    print(
        "logreg:",
        "val acc =", round(logreg_val_acc, 4),
        "test acc =", round(logreg_test_acc, 4),
        "macro F1 =", round(logreg_test_f1, 4),
        "gap =", round(logreg_gap, 4),
    )
    print(
        "lda:",
        "val acc =", round(lda_val_acc, 4),
        "test acc =", round(lda_test_acc, 4),
        "macro F1 =", round(lda_test_f1, 4),
        "gap =", round(lda_gap, 4),
    )


# 4. print a compact summary table
print("\nSummary table")
print(
    "dim | explained_var | logreg_val | logreg_test | logreg_f1 | lda_val | lda_test | lda_f1"
)
for row in results:
    print(
        f"{row['pca_dim']:>3} | "
        f"{row['explained_variance']:.4f} | "
        f"{row['logreg_val_acc']:.4f} | "
        f"{row['logreg_test_acc']:.4f} | "
        f"{row['logreg_test_f1']:.4f} | "
        f"{row['lda_val_acc']:.4f} | "
        f"{row['lda_test_acc']:.4f} | "
        f"{row['lda_test_f1']:.4f}"
    )


# 5. print confusion matrices only for the main comparison models
print("\nConfusion matrix for subsampled Logistic Regression")
print(logreg_sub_cm)

print("\nConfusion matrix for subsampled LDA")
print(lda_sub_cm)

print("\nConfusion matrix for PCA=100 Logistic Regression")
print(logreg_cm_dim100)

print("\nConfusion matrix for PCA=100 LDA")
print(lda_cm_dim100)


# 6. report-style interpretation
best_logreg = max(results, key=lambda x: x["logreg_test_acc"])
best_lda = max(results, key=lambda x: x["lda_test_acc"])

print("\nModel summary for report:")
print(
    "We compare two types of linear baselines on SVHN: "
    "a subsampled high-dimensional baseline without PCA, and PCA-based linear baselines over multiple dimensions."
)
print(
    f"For the subsampled experiment without PCA, logistic regression achieves test accuracy {logreg_sub_test_acc:.4f} "
    f"and LDA achieves test accuracy {lda_sub_test_acc:.4f}."
)
print(
    f"Across PCA dimensions, the best logistic regression test accuracy is {best_logreg['logreg_test_acc']:.4f} "
    f"at PCA dimension {best_logreg['pca_dim']}, while the best LDA test accuracy is {best_lda['lda_test_acc']:.4f} "
    f"at PCA dimension {best_lda['pca_dim']}."
)
print(
    "This comparison helps evaluate whether PCA improves generalization by reducing noise and redundancy in the original 3072-dimensional pixel space."
)
