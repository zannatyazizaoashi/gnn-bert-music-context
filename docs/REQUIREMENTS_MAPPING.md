# Requirements mapping

This document maps every item in the nine-page project brief to an implementation or output.

| Brief requirement | Implementation | Evidence/output |
|---|---|---|
| Audio at 22,050 Hz | `src/audio_features.py:load_audio` | preprocessing summary |
| 128-bin log-mel | `log_mel_spectrogram` | saved mel tensors |
| 5-10 s segmentation | 5 s default in `config.yaml` | graph JSON `num_nodes` |
| Temporal + cosine graph | `src/graph_builder.py` | 20 JSON graph samples |
| BERT tokenization, max length | Hugging Face tokenizer, 96 tokens | config/checkpoint |
| Official split | FMA `set/split` column | `results/split_audit.json` |
| Task 1 BERT | `BertTagClassifier` | BERT metrics/curves |
| Task 2 GNN | 3-layer GraphSAGE + mean pool | GNN metrics/curves |
| CNN baseline | `MelCNNClassifier` | CNN metrics/curves |
| Task 3 early concat | `EarlyConcatFusion` | ablation result |
| Task 3 cross-attention | `CrossAttentionFusion` | primary fusion result |
| Macro/Micro-F1 | `src/evaluate.py` | `results/metrics.json` |
| AUC-PR | sklearn average precision | same metrics JSON/table |
| t-SNE of fusion z | `src/visualize.py` | `plots/fusion_tsne.png` |
| Three case studies | graph plot + predictions + token attention | three PNGs and JSON |
| At least two baselines | random, CNN, BERT, GNN | comparison plot/table |
| Task 4 optional | symmetric InfoNCE dual encoder | retrieval metrics when enabled |
| 20 graph samples | automatic first-20 export | `data/processed/samples/` |
| Reproducibility | fixed seed, YAML config, checkpoints, pinned ranges | repo files |
| Demo notebook | one audio + text inference | `notebooks/demo_context.ipynb` |
| 6-10 page paper | IEEE conference source | `report/main.tex` |

## Scope interpretation

Task 4 is marked “optional -> bonus” in the grading table. It remains available through `python -m src.run_all --include-bonus`, but the mandatory path prioritizes Tasks 1-3 and report quality.

FMA-small provides audio, textual metadata, and hierarchical genre annotations. The default balanced eight-genre setting is compatible with Task 2 genre classification and produces the same one-vs-rest metrics used for Task 1/3. To run a genuine multi-label hierarchy experiment, change `data.label_mode` to `genres_all`; the label vocabulary is learned from training data only.

