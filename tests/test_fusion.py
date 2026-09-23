import torch
from torch import nn
from torch_geometric.data import Batch

import src.fusion_model as fusion_module
from src.graph_builder import make_graph


class DummyTextEncoder(nn.Module):
    def __init__(self, model_name, freeze_layers=0, dropout=0.0):
        super().__init__()
        self.output_dim = 24
        self.embedding = nn.Embedding(100, self.output_dim)

    def forward(self, tokens):
        sequence = self.embedding(tokens["input_ids"])
        return sequence[:, 0], sequence


def test_cross_attention_fusion_shapes(monkeypatch):
    monkeypatch.setattr(fusion_module, "TextEncoder", DummyTextEncoder)
    labels = torch.tensor([1.0, 0.0, 0.0])
    graphs = [make_graph(torch.randn(5, 16), labels, i, "text", 0.8, 2) for i in range(2)]
    graph_batch = Batch.from_data_list(graphs)
    tokens = {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    model = fusion_module.CrossAttentionFusion(
        graph_input_dim=16, num_labels=3, model_name="dummy",
        gnn_hidden_dim=24, gnn_layers=2, fusion_dim=24, attention_heads=4,
    )
    output = model(graph_batch, tokens)
    assert output["logits"].shape == (2, 3)
    assert output["embedding"].shape == (2, 24)
    assert output["attention"].shape == (2, 4, 8)

