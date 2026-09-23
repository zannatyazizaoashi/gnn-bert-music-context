# Final submission checklist

Deadline shown in the supplied brief: **2 October 2026**.

## Before the final run

- [ ] Add every group member's exact name, student ID, and email to `report/main.tex`.
- [ ] Keep `data.label_mode: top_genre` unless the faculty explicitly demands multi-label child genres.
- [ ] Run the `quick` profile once without errors.
- [ ] Confirm `results/split_audit.json` has zero track leakage.

## Final run

- [ ] Use a GPU runtime and `final` profile.
- [ ] Train random, CNN, BERT, GNN, early-concat, and cross-attention.
- [ ] Record the runtime/hardware in the report.
- [ ] Confirm the results file says `profile: final` for trainable models.
- [ ] Do not copy illustrative values from the faculty PDF.

## Required files

- [ ] Full source code, README, requirements, and config.
- [ ] At least 20 actual graph JSON samples.
- [ ] F1/AUC-PR comparison plot and training curves.
- [ ] Fusion t-SNE plot.
- [ ] Three case studies.
- [ ] `notebooks/demo_context.ipynb` runs end to end.
- [ ] Final report is 6-10 pages and contains real numbers.
- [ ] GitHub repository is accessible to the faculty or ZIP opens normally.

## Report quality

- [ ] Dataset size and split counts match the generated manifest.
- [ ] Audio/text/graph pipeline is described.
- [ ] Target leakage control is stated.
- [ ] All models use the same test split.
- [ ] Macro-F1, Micro-F1, and Macro AUC-PR are included.
- [ ] Ablation interpretation is honest, including negative results.
- [ ] Limitations mention weak metadata text and lack of mood labels in FMA-small.
- [ ] References include FMA, BERT/DistilBERT, GraphSAGE, and t-SNE.

## Viva preparation

- [ ] Explain why GraphSAGE is different from a CNN.
- [ ] Explain what a node and an edge mean here.
- [ ] Explain why logits—not sigmoid probabilities—go into `BCEWithLogitsLoss`.
- [ ] Explain why the classification threshold is tuned on validation data only.
- [ ] Explain early concatenation versus cross-attention.
- [ ] Explain how target-label leakage was prevented.

