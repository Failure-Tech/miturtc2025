import torch
import torch.nn as nn
import torch.nn.functional as F

# basically meant for sample-specific pathway weights and usch
class DynamicPathwayGate(nn.Module):
    def __init__(self, num_genes, num_pathways):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(num_genes, 64),
            nn.ReLU(),
            nn.Linear(64, num_pathways),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate_network(x)
    
# meant to modify attention head based on braak_stage
class BraakAwareAttention(nn.Module):
    def __init__(self, embed_dim, num_braak_stages=6):
        super().__init__()
        self.braak_embed = nn.Embedding(num_braak_stages, embed_dim)

    def forward(self, query, key, value, braak_stages):
        braak_bias = self.braak_embed(braak_stages).unsqueeze(-1)
        attn_weights = torch.matmul(query + braak_bias, key.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, value)
        
        return torch.matmul(attn_weights, value)
    
# contrastraining the attention ehad with the actual protein-protein interactions (to make it more rigourous than a simple attention head)    

class PPIMaskedAttention(nn.Module):
    def __init__(self, num_genes, embed_dim, ppi_mask):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.ppi_mask = ppi_mask  # [num_genes, num_genes], already a tensor
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: [batch_size, num_genes, embed_dim]
        Q = self.query(x)  # [batch_size, num_genes, embed_dim]
        K = self.key(x)
        V = self.value(x)

        # Compute raw attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, num_genes, num_genes]

        # Apply PPI mask: keep only valid interactions
        mask = self.ppi_mask.unsqueeze(0)  # [1, num_genes, num_genes]
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = self.softmax(attn_scores)  # [batch_size, num_genes, num_genes]

        output = torch.matmul(attn_weights, V)  # [batch_size, num_genes, embed_dim]
        return output
