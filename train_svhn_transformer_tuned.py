from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


project_root = Path(__file__).resolve().parent
data = np.load(project_root / "artifacts" / "svhn_processed_data.npz")

X_train = data["X_train_image"]
y_train = data["y_train"]
X_val = data["X_val_image"]
y_val = data["y_val"]
X_test = data["X_test_image"]
y_test = data["y_test"]


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("device:", device)


class SvhnDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ]
)

eval_transform = None


batch_size = 256
train_loader = DataLoader(
    SvhnDataset(X_train, y_train, transform=train_transform),
    batch_size=batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    SvhnDataset(X_val, y_val, transform=eval_transform),
    batch_size=batch_size,
    shuffle=False,
)
test_loader = DataLoader(
    SvhnDataset(X_test, y_test, transform=eval_transform),
    batch_size=batch_size,
    shuffle=False,
)


class TunedTransformer(nn.Module):
    def __init__(self, embed_dim=128, depth=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.patch_size = 4
        self.num_patches = (32 // 4) * (32 // 4)
        self.patch_dim = 3 * 4 * 4

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 10)

    def forward(self, x):
        patches = x.unfold(2, 4, 4).unfold(3, 4, 4)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(x.shape[0], self.num_patches, self.patch_dim)

        tokens = self.patch_embed(patches)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)
        return self.head(encoded[:, 0])


model = TunedTransformer(embed_dim=128, depth=4, num_heads=4, dropout=0.1).to(device)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

epochs = 30
patience = 6
best_val_acc = 0.0
best_state = None
best_epoch = 0
epochs_without_improvement = 0


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * xb.size(0)
        train_true.extend(yb.cpu().numpy())
        train_pred.extend(logits.argmax(dim=1).cpu().numpy())

    scheduler.step()
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
        "lr =", optimizer.param_groups[0]["lr"],
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()
        best_epoch = epoch
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("early stopping triggered")
        break


model.load_state_dict(best_state)
model.eval()


def evaluate(loader):
    true = []
    pred = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            true.extend(yb.numpy())
            pred.extend(logits.argmax(dim=1).cpu().numpy())
    return np.array(true), np.array(pred)


train_true, train_pred = evaluate(train_loader)
val_true, val_pred = evaluate(val_loader)
test_true, test_pred = evaluate(test_loader)

train_acc = accuracy_score(train_true, train_pred)
val_acc = accuracy_score(val_true, val_pred)
test_acc = accuracy_score(test_true, test_pred)
test_f1 = f1_score(test_true, test_pred, average="macro")
gap = train_acc - val_acc
cm = confusion_matrix(test_true, test_pred)

print("\nTuned Transformer")
print("best epoch:", best_epoch)
print("train accuracy:", round(train_acc, 4))
print("validation accuracy:", round(val_acc, 4))
print("test accuracy:", round(test_acc, 4))
print("test macro F1:", round(test_f1, 4))
print("generalization gap:", round(gap, 4))
print("confusion matrix:")
print(cm)

print("\nModel summary for report:")
print(
    "We train a more fully tuned transformer using AdamW, weight decay, label smoothing, "
    "cosine learning-rate scheduling, gradient clipping, validation checkpointing, and light image augmentation."
)
print(
    f"The best checkpoint is selected at epoch {best_epoch}, with validation accuracy {val_acc:.4f}, "
    f"test accuracy {test_acc:.4f}, and test macro-F1 {test_f1:.4f}."
)
