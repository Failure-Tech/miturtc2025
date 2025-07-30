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
    def __init__ (self, embed_dim, ppi_mask):
        super().__init__()
        self.ppi_mask = ppi_mask # TODO: need topreload this w a tuple (num_genes, num_genes)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return attn_out * self.ppi_mask