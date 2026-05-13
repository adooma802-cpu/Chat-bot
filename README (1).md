# 🏥 MediBot — Doctor Chatbot using Deep Learning & NLP

> A comparative study of LSTM Seq2Seq, Transformer (from scratch), and DistilBERT+Retrieval architectures for medical question answering.

---

## 📋 Project Overview

**MediBot** is an AI-powered medical chatbot trained on ~100,000 real patient-doctor conversations from the [ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) dataset. The project implements and compares three distinct deep learning architectures to answer the question: *Which architecture is best suited for a medical Q&A chatbot?*

> ⚠️ **Educational Purpose Only** — This system is not a substitute for professional medical advice. Any real medical deployment requires clinical validation and regulatory approval.

---

## 📁 Project Structure

```
adham/
├── 00_EDA_and_Preprocessing.ipynb      # Shared EDA + preprocessing pipeline
├── 01_LSTM_Seq2Seq_Fast_Final.ipynb    # Model 1: BiLSTM + Bahdanau Attention
├── 02_Transformer_Scratch_Final.ipynb  # Model 2: Transformer from scratch
├── 04_DistilBERT_v3_final.ipynb        # Model 3: DistilBERT + Retrieval
├── 05_Comparative_Analysis_Fixed.ipynb # Cross-model comparison & final report
└── 06_DoctorChatbot_GUI.ipynb          # Interactive MediBot chat GUI
```

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| **Name** | `lavita/ChatDoctor-HealthCareMagic-100k` |
| **Source** | Hugging Face Datasets |
| **Size** | ~100,000 patient-doctor conversation pairs |
| **Split** | 80% train / 10% val / 10% test |
| **Avg Input Length** | ~52 words (patient question) |
| **Avg Output Length** | ~156 words (doctor response) |

---

## 🧠 Models

### Model 1 — LSTM Seq2Seq with Bahdanau Attention
**Notebook:** `01_LSTM_Seq2Seq_Fast_Final.ipynb`

A classic encoder-decoder architecture with bidirectional LSTM and additive attention.

```
Input Tokens → Embedding → BiLSTM Encoder → Bahdanau Attention → LSTM Decoder → Output
```

| Hyperparameter | Value |
|---|---|
| Embedding dim | 128 |
| Hidden dim | 256 |
| Encoder layers | 2 (bidirectional) |
| Dropout | 0.3 |
| Effective batch size | 256 (AMP + gradient accumulation) |
| Parameters | **15.8M** |

---

### Model 2 — Transformer from Scratch
**Notebook:** `02_Transformer_Scratch_Final.ipynb`

Full encoder-decoder Transformer (Vaswani et al., 2017) built from scratch with multi-head self-attention and positional encoding.

```
Input → Embedding + PosEnc → 3×Encoder (MH-Attention + FFN) → 3×Decoder (Masked + Cross-Attn + FFN) → Output
```

| Hyperparameter | Value |
|---|---|
| d_model | 256 |
| Attention heads | 8 |
| Encoder/Decoder layers | 3 each |
| FFN dimension | 512 |
| Max sequence length | 128 |
| Parameters | **18.2M** |

---

### Model 3 — DistilBERT + Retrieval-Augmented Generation
**Notebook:** `04_DistilBERT_v3_final.ipynb`

Two-stage pipeline: fine-tune DistilBERT for medical topic classification, then use CLS embeddings for semantic similarity retrieval (FAISS).

```
Patient Query → DistilBERT (6 layers, 768-dim) → [CLS] → Classifier (10 topics) + FAISS Retrieval
```

| Hyperparameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Max length | 256 tokens |
| Classification heads | 10 medical topics |
| Learning rate | 2e-5 |
| Parameters | **66.8M** |

---

## 📊 Results

| Model | BLEU-1 | BLEU-2 | BLEU-4 | Val PPL | Epochs |
|---|---|---|---|---|---|
| LSTM Seq2Seq | 18.4 | 9.2 | 3.1 | 28.5 | 12 |
| Transformer (Scratch) | 22.1 | 11.8 | 4.6 | 21.3 | 11 |
| **DistilBERT + Retrieval** | **29.4** | **18.6** | **9.1** | — | **5** |

**🥇 Best overall:** DistilBERT + Retrieval (BLEU-4: 9.1)  
**🥈 Best from-scratch:** Transformer (BLEU-4: 4.6, PPL: 21.3)  
**🥉 Most efficient:** LSTM Seq2Seq (15.8M params)

---

## 🖥️ Interactive GUI

**Notebook:** `06_DoctorChatbot_GUI.ipynb`

Renders a full MediBot chat interface inside Google Colab. Features:
- Switch between LSTM, Transformer, and DistilBERT live
- **⚡ Compare All** — runs all 3 models side-by-side
- Auto-detects medical topic (10 specialty categories)
- Displays BLEU metrics with animated confidence bars

### Setup

```python
# In Notebook 06, set your Anthropic API key via Colab Secrets:
# Left sidebar → 🔑 Secrets → Add "ANTHROPIC_API_KEY"
# Then run the GUI cell — MediBot renders in the output
```

---

## 🚀 Getting Started

### Requirements

```bash
pip install torch pandas numpy matplotlib scikit-learn nltk datasets \
            transformers faiss-cpu rouge-score tqdm
```

### Run Order

```
1. 00_EDA_and_Preprocessing.ipynb   ← Run first (creates data/ splits & vocab)
2. 01_LSTM_...ipynb                 ← Train LSTM model
3. 02_Transformer_...ipynb          ← Train Transformer model
4. 04_DistilBERT_...ipynb           ← Fine-tune DistilBERT
5. 05_Comparative_Analysis_...ipynb ← Compare all results
6. 06_DoctorChatbot_GUI.ipynb       ← Launch interactive demo
```

> **Note:** All notebooks auto-download the dataset from Hugging Face if `data/train.csv` is not found. A GPU (e.g., Google Colab T4) is strongly recommended.

### Generated Files

After running all notebooks:

```
data/
├── train.csv              # 80% training split
├── val.csv                # 10% validation split
├── test.csv               # 10% test split
└── vocab.json             # Word-to-index mapping

lstm_seq2seq_best.pt       # Saved LSTM weights
transformer_best.pt        # Saved Transformer weights
distilbert_med_best.pt     # Saved DistilBERT weights

data/results_lstm.json         # LSTM evaluation results
data/results_transformer.json  # Transformer evaluation results
data/results_distilbert.json   # DistilBERT evaluation results
```

---

## 🔑 Key Findings

1. **Transfer learning dominates** — DistilBERT converges in 5 epochs vs. 11–12 for from-scratch models.
2. **Retrieval augmentation wins on BLEU** — Retrieved real doctor answers maximize n-gram overlap and minimize hallucination.
3. **Self-attention > recurrence** — The Transformer outperforms the LSTM despite similar parameter counts, confirming Vaswani et al. (2017).
4. **BLEU has limits** — Diverse but medically correct responses may score lower; future work should use BERTScore.
5. **Error patterns are architecture-specific** — LSTM truncates; Transformer misses urgency; DistilBERT fails on novel queries.

---

## 🏗️ Architecture Comparison

| Dimension | LSTM Seq2Seq | Transformer | DistilBERT+RAG |
|---|---|---|---|
| Long-range context | ⚠️ Limited | ✅ Excellent | ✅ Excellent |
| Novel query handling | ✅ Good | ✅ Good | ⚠️ Limited |
| Factual reliability | ❌ Lowest | ⚠️ Moderate | ✅ Highest |
| Training speed | ✅ Fastest | ✅ Fast | ⚠️ Slowest |
| Deployment cost | ✅ Lowest | ✅ Low | ⚠️ Highest |

---

## 📚 References

- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- Sanh et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108.
- Bahdanau et al. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.
- Li et al. (2023). *ChatDoctor: A Medical Chat Model Fine-Tuned on LLaMA.* arXiv:2303.14070.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. The MediBot chatbot is not a licensed medical device and must not be used as a substitute for professional medical consultation, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.
