import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

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
    data_path = "processsed.csv"  # <-- update this to your CSV file

    # Load dataset
    try:
        dataset = GeneExpressionData(data_path, num_genes)
    except Exception as e:
        print("Failed to load dataset.")
        exit()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and training components
    model = GeneAttentionNet(num_genes=num_genes)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss = []

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        train_loss.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(train_loss, label="Training Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.legend()
    plt.savefig("test_fig.png")
    plt.show()
