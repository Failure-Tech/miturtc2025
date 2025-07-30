import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import DynamicPathwayGate, BraakAwareAttention, PPIMaskedAttention

class GeneAttentionNet(nn.Module):
    def __init__(self, num_genes, pathway_masks, ppi_mask):
        super().__init__()

        self.embedding = nn.Linear(num_genes, 128) # using a simple projection lnayer first

        # 2 attention ehads, ad genes head and data-driven head
        self.dynamic_gate = DynamicPathwayGate(num_genes, len(pathway_masks))
        self.braak_attention = BraakAwareAttention(128)
        self.graph_attention = PPIMaskedAttention(128, ppi_mask)

        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=128, num_heads=1), # ad heads
            nn.MultiheadAttention(embed_dim=128, num_heads=1), # data-drivien head
            self.graph_attention # new ppi constrained attneiton ehad
        ])

        self.pathway_scorers = nn.ModuleList([
            nn.Linear(128, 1) for name in pathway_masks.keys()
        ])

        # classifying them
        self.classifier = nn.Sequential(
            nn.Linear(128 + len(pathway_masks), 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, braak_stages=None):
        x = self.embedding(x)

        pathway_weights = self.dynamic_gate(x)

        attn_outputs = []
        for i, head in enumerate(self.attention_heads):
            if i == 0:
                attn_out, _ = head(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            elif i == 1 and braak_stages is not None:
                attn_out = self.braak_attention(x, x, x, braak_stages)

            else:
                attn_out = head(x.unsqueeze(0)).unsqueeze(0)
            
            attn_outputs.append(attn_out)

            x = torch.mean(torch.stack(attn_outputs), dim=0)

            pathway_scores = []
            for i, (name, scorer) in enumerate(self.pathway_scorers.items()):
                masked = x * pathway_weights[:, i].unsqueeze(1)
                score = scorer(masked)
                pathway_scores.append(score)

            combined = torch.cat([x] + pathway_scores, dim=1)
            logits = self.classifier(combined)

            return logits.unsqueeze(-1), pathway_weights, x