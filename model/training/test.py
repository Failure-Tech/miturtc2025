import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import random_split
import os
import pandas as pd
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        qkv = self.qkv_proj(x)  # [B, T, 3 * E]
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3)  # [H, B, T, 3*D]

        q, k, v = qkv.chunk(3, dim=-1)  # Each is [H, B, T, D]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [H, B, T, T]
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = attn_probs @ v  # [H, B, T, D]

        attn_output = attn_output.permute(1, 2, 0, 3).reshape(batch_size, seq_len, embed_dim)  # [B, T, E]
        return self.out_proj(attn_output)

class GeneAttentionNet(nn.Module):
    def __init__(self, num_genes, embed_dim=128, num_heads=4):
        super().__init__()
        self.num_genes = num_genes
        self.embed_dim = embed_dim

        # Project the gene vector into an embedding
        self.gene_embedding = nn.Linear(num_genes, embed_dim)

        # Project to sequence of embeddings (fake sequence length = num_patches = 4 for example)
        self.tokenizer = nn.Linear(embed_dim, embed_dim * 4)

        # Custom Multi-Head Attention
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)  # Binary classification (logit)
        )

    def forward(self, x):
        """
        Input: x shape [batch_size, num_genes]
        """
        x = self.gene_embedding(x)  # [B, E]
        x = self.tokenizer(x).reshape(x.size(0), 4, self.embed_dim)  # [B, 4, E]

        attn_out = self.attention(x)  # [B, 4, E]
        pooled = attn_out.mean(dim=1)  # [B, E] - mean pooling over tokens

        logits = self.classifier(pooled)  # [B, 1]
        return logits
    
def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()

            total += labels.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        return val_loss, val_acc

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from .datasets import GeneExpressionData  # Update path as needed

    # Hyperparameters
    num_genes = 1247
    batch_size = 32
    epochs = 100
    lr = 1e-4
    val_split = 0.2
    data_path = "processsed.csv"  # <-- update this to your CSV file

    # Load dataset
    try:
        dataset = GeneExpressionData(data_path, num_genes)
    except Exception as e:
        print("Failed to load dataset.")
        exit()

    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and training components
    model = GeneAttentionNet(num_genes=num_genes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, validation_losses = [], []
    train_accuracy, validation_accuracy = [], []

    # Training loop
    for epoch in range(epochs):
        train_loss = 0.0
        correct = 0
        total = 0

        model.train()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # batch_y = batch_y.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.7) # remove if not necessary
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == batch_y).sum().item()

            total += batch_x.size(0)

        train_loss = train_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracy.append(train_acc)
        validation_losses.append(val_loss)
        validation_accuracy.append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    df = pd.DataFrame({
        "Epoch": list(range(1, epochs+1)),
        "Training Loss": train_losses,
        "Validation Loss": validation_losses,
        "Training Accuracy": train_accuracy,
        "Validation Accuracy": validation_accuracy
    })

    df.to_csv("training_log.csv", index=False)

    plt.figure(figsize=(10, 4))

    plt.plot(train_losses, label="Training Losses")
    plt.plot(validation_losses, label="Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    plt.show()

    plt.plot(train_accuracy, label="Training Accuracy")
    plt.plot(validation_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.tight_layout()
    plt.savefig("accuracy_plot.png")
    plt.show()