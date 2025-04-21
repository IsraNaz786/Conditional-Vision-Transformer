import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image
from collections import Counter


# Dataset Class
class GTSRB_Dataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.image_paths = self.df['Path'].values
        self.labels = self.df['ClassId'].values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_name).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Dataset Preparation with Caching
def prepare_gtsrb_dataset(data_root, train_csv, test_csv, transform):
    cache_dir = os.path.join(os.getcwd(), "cached_datasets")
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train.pt")
    val_cache = os.path.join(cache_dir, "val.pt")
    test_cache = os.path.join(cache_dir, "test.pt")

    if os.path.exists(train_cache) and os.path.exists(val_cache) and os.path.exists(test_cache):
        print("Loading dataset from cache...")
        train_data = torch.load(train_cache)
        val_data = torch.load(val_cache)
        test_data = torch.load(test_cache)
    else:
        print("Processing and caching dataset...")
        full_train_dataset = GTSRB_Dataset(root_dir=data_root, csv_file=train_csv, transform=transform)
        test_dataset = GTSRB_Dataset(root_dir=data_root, csv_file=test_csv, transform=transform)

        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_data, val_data = random_split(full_train_dataset, [train_size, val_size])

        torch.save(train_data, train_cache)
        torch.save(val_data, val_cache)
        torch.save(test_dataset, test_cache)

        test_data = test_dataset

    return train_data, val_data, test_data


def get_sampler(dataset):
    targets = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(targets)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.to(x.device)
        return x


# Gating Network
class GatingNetwork(nn.Module):
    def __init__(self, embed_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        cls_token = x[:, 0]
        return self.softmax(self.fc(cls_token))


# Conditional Attention
class ConditionalMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_experts):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.head_dim = embed_dim // num_heads

        self.q_experts = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_experts)])
        self.k_experts = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_experts)])
        self.v_experts = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_experts)])
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, gating_weights):
        B, N, D = x.shape
        device = x.device
        Q = sum(w.unsqueeze(-1).unsqueeze(-1).to(device) * proj(x) for w, proj in zip(gating_weights.t(), self.q_experts))
        K = sum(w.unsqueeze(-1).unsqueeze(-1).to(device) * proj(x) for w, proj in zip(gating_weights.t(), self.k_experts))
        V = sum(w.unsqueeze(-1).unsqueeze(-1).to(device) * proj(x) for w, proj in zip(gating_weights.t(), self.v_experts))

        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = attn @ V
        context = context.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(context), attn


# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_experts, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ConditionalMultiheadAttention(embed_dim, num_heads, num_experts)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x, gating_weights):
        attn_out, attn_weights = self.attn(self.norm1(x), gating_weights)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


# Full Conditional ViT Model
class ConditionalViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256, num_heads=4, num_layers=4,
                 num_classes=43, num_experts=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.gating = GatingNetwork(embed_dim, num_experts)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, num_experts)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        gating_weights = self.gating(x)
        attn_maps = []
        for layer in self.encoder_layers:
            x, attn = layer(x, gating_weights)
            attn_maps.append(attn)
        x = self.norm(x[:, 0])
        return self.head(x), attn_maps


# Main Function
def main():
    data_root = r"F:\\PHD Work 2025\\Second Paper Code\\data\\archive"
    train_csv = os.path.join(data_root, 'Train.csv')
    test_csv = os.path.join(data_root, 'Test.csv')

    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_data, val_data, test_data = prepare_gtsrb_dataset(data_root, train_csv, test_csv, transform_train)
    sampler = get_sampler(train_data)

    train_loader = DataLoader(train_data, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = ConditionalViT()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0
    for epoch in range(50):
        model.train()
        correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    all_preds, all_labels, max_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            probs = F.softmax(outputs, dim=1)
            max_prob, preds = torch.max(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            max_probs.extend(max_prob.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    with open("predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TrueLabel", "PredictedLabel", "MaxProbability"])
        for t, p, prob in zip(all_labels, all_preds, max_probs):
            writer.writerow([t, p, f"{prob:.4f}"])


if __name__ == "__main__":
    main()
