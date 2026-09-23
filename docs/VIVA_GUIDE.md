# Short viva guide

**What is the project doing?**  It predicts musical genre context by combining the sound structure of a track with textual metadata. The audio is represented as a graph of time segments; DistilBERT represents title/album/artist text.

**What is one graph node?**  A five-second audio segment summarized by log-mel, chroma, MFCC, and spectral statistics.

**What is one graph edge?**  Either two adjacent segments or two non-adjacent segments whose normalized feature vectors have cosine similarity at least 0.80.

**Why GraphSAGE?**  It aggregates neighboring segment representations, allowing repeated or related song sections to exchange information before mean pooling creates one track vector.

**Why BERT?**  BERT creates contextual token representations, so a word is interpreted with the surrounding title/album/artist text rather than as an isolated bag-of-words feature.

**What is early concatenation?**  Independently pool one graph vector and one BERT CLS vector, concatenate them, then classify.

**What is cross-attention?**  The graph vector becomes a query over all BERT token vectors. The model can emphasize text tokens most useful for the current audio structure.

**Why BCEWithLogitsLoss?**  Each label is treated as a separate binary decision. This loss combines a stable sigmoid calculation with binary cross-entropy.

**Why tune a threshold?**  A fixed 0.5 threshold is not always best under label imbalance. The code selects one threshold on validation data and applies it once to the untouched test set.

**How is leakage avoided?**  Target genres are not placed in the BERT sentence. Text contains only track title, album title, and artist name. The official FMA split is used, and track/artist overlap is audited.

**What is the fairest ablation?**  All models use the same processed tracks, labels, and official split. Removing text yields GNN-only; removing graph yields BERT-only; changing only fusion compares concatenation with cross-attention.

**What is the main limitation?**  FMA metadata text is much shorter and weaker than full lyrics or expert captions, and FMA-small has genres rather than explicit mood/valence labels. This makes the experiment practical and reproducible but limits semantic richness.

