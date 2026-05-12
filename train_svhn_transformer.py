from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset


# 1. read processed SVHN image data
project_root = Path(__file__).resolve().parent
data = np.load(project_root / "artifacts" / "svhn_processed_data.npz")

X_train = data["X_train_image"]
y_train = data["y_train"]
X_val = data["X_val_image"]
y_val = data["y_val"]
X_test = data["X_test_image"]
y_test = data["y_test"]


# 2. choose device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("device:", device)


# 3. convert numpy arrays to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
y_test = torch.tensor(y_test, dtype=torch.long)


# 4. data loaders
batch_size = 256
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)


# 5. transformer model
class SmallTransformer(nn.Module):
    def __init__(self, embed_dim, depth):
        super().__init__()
        self.patch_size = 4
        self.num_patches = (32 // 4) * (32 // 4)
        self.patch_dim = 3 * 4 * 4
        self.embed_dim = embed_dim

        self.patch_embed = nn.Linear(self.patch_dim, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=self.embed_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(self.embed_dim, 10)

    def forward(self, x):
        patches = x.unfold(2, 4, 4).unfold(3, 4, 4)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(x.shape[0], self.num_patches, self.patch_dim)

        tokens = self.patch_embed(patches)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed

        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])


# 6. train and compare several transformer settings
settings = [
    {"name": "Transformer depth=2 embed_dim=64", "depth": 2, "embed_dim": 64},
    {"name": "Transformer depth=4 embed_dim=64", "depth": 4, "embed_dim": 64},
    {"name": "Transformer depth=2 embed_dim=128", "depth": 2, "embed_dim": 128},
    {"name": "Transformer depth=4 embed_dim=128", "depth": 4, "embed_dim": 128},
]

epochs = 5
all_results = []

for setting in settings:
    print("\n" + "=" * 80)
    print(setting["name"])

    model = SmallTransformer(embed_dim=setting["embed_dim"], depth=setting["depth"]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_true = []
        train_pred = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            train_true.extend(yb.cpu().numpy())
            train_pred.extend(logits.argmax(dim=1).cpu().numpy())

        train_acc = accuracy_score(train_true, train_pred)

        model.eval()
        val_true = []
        val_pred = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                val_true.extend(yb.numpy())
                val_pred.extend(logits.argmax(dim=1).cpu().numpy())

        val_acc = accuracy_score(val_true, val_pred)
        print(
            f"epoch {epoch}:",
            "train acc =", round(train_acc, 4),
            "val acc =", round(val_acc, 4),
            "train loss =", round(train_loss / len(train_loader.dataset), 4),
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    model.eval()

    train_true = []
    train_pred = []
    with torch.no_grad():
        for xb, yb in train_loader:
            xb = xb.to(device)
            logits = model(xb)
            train_true.extend(yb.numpy())
            train_pred.extend(logits.argmax(dim=1).cpu().numpy())

    val_true = []
    val_pred = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            val_true.extend(yb.numpy())
            val_pred.extend(logits.argmax(dim=1).cpu().numpy())

    test_true = []
    test_pred = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            test_true.extend(yb.numpy())
            test_pred.extend(logits.argmax(dim=1).cpu().numpy())

    train_acc = accuracy_score(train_true, train_pred)
    val_acc = accuracy_score(val_true, val_pred)
    test_acc = accuracy_score(test_true, test_pred)
    test_f1 = f1_score(test_true, test_pred, average="macro")
    gap = train_acc - val_acc
    cm = confusion_matrix(test_true, test_pred)

    print("\n" + setting["name"])
    print("train accuracy:", round(train_acc, 4))
    print("validation accuracy:", round(val_acc, 4))
    print("test accuracy:", round(test_acc, 4))
    print("test macro F1:", round(test_f1, 4))
    print("generalization gap:", round(gap, 4))
    print("confusion matrix:")
    print(cm)

    all_results.append(
        {
            "name": setting["name"],
            "depth": setting["depth"],
            "embed_dim": setting["embed_dim"],
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "gap": gap,
            "cm": cm,
        }
    )


# 7. print summary table
print("\nSummary table")
print("model | val acc | test acc | macro F1 | gap")
for row in all_results:
    print(
        f"{row['name']} | "
        f"{row['val_acc']:.4f} | "
        f"{row['test_acc']:.4f} | "
        f"{row['test_f1']:.4f} | "
        f"{row['gap']:.4f}"
    )


# 8. report summary
best_model = max(all_results, key=lambda x: x["test_acc"])
print("\nModel summary for report:")
print(
    "We compare four small vision-transformer-style models on normalized SVHN image tensors "
    "to study the effect of transformer depth and embedding width."
)
print(
    f"The best transformer result is achieved by {best_model['name']}, "
    f"with validation accuracy {best_model['val_acc']:.4f}, "
    f"test accuracy {best_model['test_acc']:.4f}, and test macro-F1 {best_model['test_f1']:.4f}."
)
print(
    "This experiment allows us to examine how increasing depth or embedding dimension changes predictive performance, "
    "generalization gap, and class-level confusion relative to the linear baselines."
)
