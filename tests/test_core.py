import numpy as np
import torch
from torch_geometric.data import Batch

from src.audio_features import AudioFeatureConfig, log_mel_spectrogram, segment_features
from src.cnn_model import MelCNNClassifier
from src.evaluate import classification_metrics, tune_threshold
from src.gnn_model import GNNClassifier
from src.graph_builder import build_edge_index, make_graph


def test_audio_feature_shapes_and_finite_values():
    config = AudioFeatureConfig(duration_seconds=10.0, segment_seconds=5.0)
    time = np.arange(int(config.sample_rate * 10.0)) / config.sample_rate
    audio = (0.5 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    nodes = segment_features(audio, config)
    mel = log_mel_spectrogram(audio, config)
    assert nodes.shape == (2, 330)
    assert mel.shape[0] == 128
    assert np.isfinite(nodes).all() and np.isfinite(mel).all()


def test_graph_has_bidirectional_temporal_edges():
    nodes = torch.randn(4, 330)
    edge_index, edge_attr = build_edge_index(nodes, similarity_threshold=2.0)
    edges = {tuple(edge) for edge in edge_index.T.tolist()}
    for source in range(3):
        assert (source, source + 1) in edges
        assert (source + 1, source) in edges
    assert edge_attr.shape[0] == edge_index.shape[1]


def test_cnn_and_gnn_output_shapes():
    labels = torch.tensor([1.0, 0.0, 0.0])
    graph_a = make_graph(torch.randn(5, 330), labels, 1, "a", 0.8, 2)
    graph_b = make_graph(torch.randn(6, 330), labels, 2, "b", 0.8, 2)
    batch = Batch.from_data_list([graph_a, graph_b])
    gnn = GNNClassifier(330, 3, hidden_dim=32, layers=2)
    cnn = MelCNNClassifier(3)
    assert gnn(graph=batch)["logits"].shape == (2, 3)
    assert cnn(mel=torch.randn(2, 1, 128, 200))["logits"].shape == (2, 3)


def test_metrics_are_bounded_and_threshold_uses_validation():
    targets = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.3], [0.1, 0.7]])
    threshold = tune_threshold(targets, probabilities)
    metrics = classification_metrics(targets, probabilities, threshold)
    assert 0.1 <= threshold <= 0.9
    for key in ("macro_f1", "micro_f1", "macro_auc_pr", "micro_auc_pr", "exact_match"):
        assert 0.0 <= metrics[key] <= 1.0

