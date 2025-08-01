import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import DynamicPathwayGate, BraakAwareAttention, PPIMaskedAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GeneAttentionNet(nn.Module):
    def __init__(self, num_genes, pathway_masks, ppi_mask):
        super().__init__()
        self.num_genes = num_genes
        
        # Embedding layer
        self.embedding = nn.Linear(num_genes, 128)
        
        # Dynamic pathway gate
        self.dynamic_gate = DynamicPathwayGate(num_genes, len(pathway_masks))
        
        # PPI projection layers
        self.ppi_proj = nn.Linear(num_genes, self.num_genes*128)  # To gene space
        self.ppi_proj_inv = nn.Linear(128, 128)  # Back to embedding space
        
        # Attention heads
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=128, num_heads=1),
            nn.MultiheadAttention(embed_dim=128, num_heads=1),
            PPIMaskedAttention(num_genes=num_genes, embed_dim=128, ppi_mask=ppi_mask)  # Operates in gene space
        ])
        
        # 4. Pathway scoring
        self.pathway_scorers = nn.ModuleList([
            nn.Linear(128, 1) for _ in pathway_masks.keys()
        ])
        
        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + len(pathway_masks), 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, braak_stages=None):
        # Input shape: [batch_size, num_genes]
        inputs = x
        pathway_weights = self.dynamic_gate(x)  # [batch_size, num_pathways]
        if torch.isnan(pathway_weights).any():
            print("NaN with pahtwayweights", flush=True)
        
        x = self.embedding(x)  # [batch_size, 128]
        if torch.isnan(x).any():
            print("NaN with embedding output", flush=True)
        
        # Get pathway weights
        
        # Process through attention heads
        attn_outputs = []
        x_reshaped = x.unsqueeze(0)  # [1, batch_size, 128] for attention
        
        for i, head in enumerate(self.attention_heads):
            if i == 0:  # Standard MultiheadAttention
                attn_out, _ = head(x_reshaped, x_reshaped, x_reshaped)
            elif i == 1:  # BraakAwareAttention
                if braak_stages is not None:
                    attn_out = self.braak_attention(x_reshaped, x_reshaped, x_reshaped, braak_stages)
                else:
                    attn_out = x_reshaped
            else:  # PPIMaskedAttention (i == 2)
                # Project to gene space for PPI mask
                # x_proj = self.ppi_proj(inputs)  # [batch_size, num_genes * 128]
                # x_proj = x_proj.view(-1, self.num_genes, 128)  # [batch_size, num_genes, 128]
                # attn_out = head(x_proj)  # [batch_size, num_genes, 128]
                # attn_out = torch.mean(attn_out, dim=1).unsqueeze(0)  # [1, batch_size, 128]

                # attn_out = self.ppi_proj_inv(attn_out)  # [1, batch_size, 128]
                # inputs shape: [batch_size, num_genes]
                x_proj = self.ppi_proj(inputs)  # [batch_size, num_genes * 128]
                if torch.isnan(x_proj).any():
                    print("NaN in ppi_proj output", flush=True)
                x_proj = x_proj.view(-1, self.num_genes, 128)  # [batch_size, num_genes, 128]
                if torch.isnan(x_proj).any():
                    print("NaN after reshaping PPI", flush=True)

                attn_out = head(x_proj)  # Expect [batch_size, num_genes, 128]
                if torch.isnan(attn_out).any():
                    print("NaN in PPI attention head", flush=True)

                attn_out = torch.mean(attn_out, dim=1)  # Average over genes -> [batch_size, 128]

                attn_out = self.ppi_proj_inv(attn_out)  # [batch_size, 128]

                attn_out = attn_out.unsqueeze(0)  # [1, batch_size, 128] to match other heads

            
            attn_outputs.append(attn_out)  # [batch_size, 128]
        
        # Combine attention outputs
        x = torch.mean(torch.stack(attn_outputs), dim=0).squeeze(0)  # [batch_size, 128]
        
        # Pathway scoring
        pathway_scores = []
        for i, scorer in enumerate(self.pathway_scorers):
            masked = x * pathway_weights[:, i].unsqueeze(-1)
            pathway_scores.append(scorer(masked))
        
        # Final classification
        stacked_pathway_scores = torch.cat(pathway_scores, dim=1)
        combined = torch.cat([x, stacked_pathway_scores], dim=1)
        logits = self.classifier(combined)
        
        return logits, pathway_weights, x