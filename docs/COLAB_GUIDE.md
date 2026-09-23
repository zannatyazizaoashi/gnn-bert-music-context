# Google Colab guide

## Recommended order

1. Open `notebooks/CSE425_Full_Project_Colab.ipynb`.
2. Choose a T4 GPU.
3. Upload the project ZIP when prompted.
4. Run the environment and GPU checks.
5. Download FMA-small and metadata. The audio archive is several GB; keep enough runtime disk space.
6. Run preprocessing with `quick` first.
7. Run all six experiments with `quick` and inspect outputs.
8. Delete `data/processed/manifest.csv`, `data/processed/graphs/`, and `data/processed/mels/` only if you need to change feature settings. Otherwise preprocessing resumes from cached files.
9. Change the profile to `final`, rerun preprocessing, then training.
10. Run export and download `submission_bundle.zip`.

## Estimated strategy

- `smoke`: 48 tracks, one epoch. Verifies code paths only; metrics are meaningless.
- `quick`: 1,000 tracks, five epochs. Useful for debugging and presentation preparation.
- `final`: all available FMA-small tracks, up to 20 epochs with early stopping. Use these results in the report.

Exact time depends on Colab GPU, download speed, and how many FMA files decode successfully. Preprocessing is CPU-heavy, while BERT/fusion training benefits most from the GPU.

## If Colab disconnects

Mount Google Drive and periodically copy these folders:

```python
!cp -r results /content/drive/MyDrive/CSE425_backup/
!cp -r plots /content/drive/MyDrive/CSE425_backup/
!cp data/processed/manifest.csv /content/drive/MyDrive/CSE425_backup/
```

Do not repeatedly copy the raw 8 GB archive or all processed tensors unless your Drive has enough space.

## Common failures

- **CUDA out of memory:** reduce `bert_batch_size` from 12 to 6 in `config.yaml`.
- **A few MP3 failures:** expected for damaged/very short FMA files; inspect `preprocessing_failures.json`.
- **No graph samples:** preprocessing did not complete 20 tracks.
- **Missing report PDF:** Colab may not have LaTeX. Upload `report/` to Overleaf using the IEEE Conference template or install `texlive-latex-extra`.
- **AUC warning:** a label may be absent in a tiny smoke split. This disappears in the final balanced run.

