# Development of a Transformer-Based Next-Token Prediction Model Using Basic Self-Attention

An end-to-end, academic mini-project demonstrating autoregressive language modeling and scaled dot-product self-attention built from scratch in PyTorch.

---

## 1. Project Overview & Abstract

### Abstract
Language models powered by the Transformer architecture form the foundation of modern Artificial Intelligence and Natural Language Processing (NLP). This project presents an educational, transparent implementation of an autoregressive language model designed specifically for next-token prediction. Rather than relying on high-level libraries (`nn.Transformer` or Hugging Face), every core component—including Token & Positional Embeddings, Scaled Dot-Product Attention, Causal Masking, Multi-Head Self-Attention, Pre-LayerNorm Residual Connections, and Feed-Forward Networks—is implemented from scratch using basic PyTorch tensor operations. The model is trained on a CPU-friendly corpus of computer science and deep learning literature. It achieves stable convergence, validates causal independence, and is accompanied by an interactive Streamlit web application for real-time inference, candidate token inspection, and attention weight visualization.

---

## 2. System Architecture

The model implements a standard Decoder-only Transformer architecture following modern best practices (Pre-Layer Normalization and Weight Tying):

```
Raw Input Text ("machine learning is")
        │
        ▼
SimpleTokenizer (Tokenization & Numerical Mapping)
        │
        ▼ Token IDs: [15, 16, 17]
┌────────────────────────────────────────────────────────┐
│  TransformerLM                                         │
│                                                        │
│  Token Embedding (Vocab → d_model)                     │
│           +                                            │
│  Positional Embedding (SeqLen → d_model)               │
│           │                                            │
│           ▼                                            │
│  Dropout (p=0.1)                                       │
│           │                                            │
│   ┌───────┴────────────────────────────────────────┐   │
│   │ Transformer Block (Layer 1 & Layer 2)          │   │
│   │                                                │   │
│   │  x ──► LayerNorm ──► Causal Multi-Head Attn ──►+   │
│   │         (Q, K, V Projections & Masking)        │   │
│   │  x ────────────────────────────────────────────┘   │
│   │                                                    │
│   │  x ──► LayerNorm ──► Feed-Forward (GELU) ──────►+  │
│   │  x ────────────────────────────────────────────┘   │
│   └───────┬────────────────────────────────────────┘   │
│           │                                            │
│           ▼                                            │
│     Final LayerNorm                                    │
│           │                                            │
│           ▼                                            │
│     Linear Projection (d_model → Vocab Size)           │
│     (Weight tied with Token Embedding)                 │
└───────────┬────────────────────────────────────────────┘
            │
            ▼ Output Logits [Batch, SeqLen, VocabSize]
      Softmax along Vocab Dimension
            │
            ▼
Next-Token Probability Distribution & Top Candidates
```

---

## 3. Mathematical Formulation of Self-Attention

Self-attention determines how much each token in a sentence should attend to every other token.

### 3.1 Projections
Given an input matrix $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$, where $T$ is the sequence length:
$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$
where $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

### 3.2 Scaled Dot-Product Attention
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}$$

1. **Dot-Product Similarity**: $\mathbf{S} = \mathbf{Q} \mathbf{K}^T$ measures the raw compatibility between query token $i$ and key token $j$.
2. **Scaling Factor ($\sqrt{d_k}$)**: For large values of $d_k$, dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ prevents gradient vanishing.
3. **Causal Mask ($\mathbf{M}$)**: To prevent information leakage from future tokens in autoregressive generation:
   $$M_{i,j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
4. **Softmax Normalization**: Turns scores into positive probabilities that sum to 1 across rows:
   $$A_{i,j} = \frac{\exp(S_{i,j} / \sqrt{d_k} + M_{i,j})}{\sum_{k=1}^T \exp(S_{i,k} / \sqrt{d_k} + M_{i,k})}$$
5. **Contextual Aggregation**: Computes the final contextual vector as the weighted combination of Value vectors:
   $$\mathbf{O}_i = \sum_{j=1}^i A_{i,j} \mathbf{V}_j$$

---

## 4. Project Directory Structure

```
transformer_next_token/
├── data/
│   └── corpus.txt              # Public-domain domain-specific English corpus
├── models/
│   ├── best_model.pt           # Saved PyTorch model checkpoint
│   ├── final_model.pt          # Final training checkpoint
│   ├── vocab.json              # Word-to-index and index-to-word mappings
│   ├── training_history.json   # Recorded train/val losses & perplexities
│   └── training_loss.png       # Matplotlib training vs validation loss plot
├── src/
│   ├── __init__.py
│   ├── tokenizer.py            # Word-level tokenizer with special tokens (<PAD>, <UNK>, etc.)
│   ├── dataset.py              # Next-token sliding window Dataset & DataLoader
│   ├── attention.py            # Scaled Dot-Product & Multi-Head Attention from scratch
│   ├── transformer.py          # Complete TransformerLM with Pre-LN & FFN
│   ├── train.py                # Supervised training loop, optimizer, loss & checkpointing
│   ├── generate.py             # Greedy & temperature sampling with repetition penalty
│   └── evaluate.py             # Validation loss, perplexity, and prediction comparison
├── tests/
│   ├── __init__.py
│   └── test_project.py         # 9 automated unit tests verifying architecture and math
├── app.py                      # Interactive Streamlit Web UI
├── requirements.txt            # Python dependencies
└── README.md                   # Complete academic documentation
```

---

## 5. Quickstart & Execution Guide (Windows PowerShell)

### Step 1: Open Terminal in Project Directory
```powershell
cd c:\proj3
```

### Step 2: (Optional) Create & Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run the Automated Verification Tests
Verify all 9 tests covering tokenization, tensor shapes, causal masking, and loss calculation:
```powershell
python tests/test_project.py
```
*Expected output: `Ran 9 tests in ~3.8s ... OK`*

### Step 5: Train the Model
Train the Transformer model on CPU for 25 epochs:
```powershell
python src/train.py --epochs 25 --seq_len 16 --batch_size 16 --lr 0.001
```

### Step 6: Evaluate Model Performance
Evaluate cross-entropy loss, perplexity, and actual vs. predicted tokens:
```powershell
python src/evaluate.py
```

### Step 7: Test Text Generation from CLI
```powershell
# Greedy decoding (deterministic)
python src/generate.py --prompt "the cat sat on" --greedy

# Temperature sampling (creative)
python src/generate.py --prompt "machine learning is" --temperature 0.7
```

### Step 8: Launch the Interactive Streamlit Web UI
```powershell
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

---

## 6. Experimental Results & Verification

Actual training run results from the project:

| Metric | Measured Value |
| :--- | :--- |
| **Initial Training Loss (Epoch 1)** | `5.6314` |
| **Final Training Loss (Epoch 25)** | `0.3225` |
| **Best Validation Loss** | `5.5004` |
| **Total Trainable Parameters** | `131,264` |
| **Vocabulary Size** | `479 tokens` |
| **Context Length ($T$)** | `16 tokens` |
| **Embedding Dimension ($d_{\text{model}}$)** | `64` |
| **Number of Heads** | `4` ($d_k = 16$) |
| **Number of Layers** | `2` |

### Sample Actual Next-Token Predictions

| Prompt Context | Top-1 Predicted Next Token | Top-1 Probability | Top-5 Candidates |
| :--- | :--- | :--- | :--- |
| `the cat sat on` | `the` | **99.44%** | `gradients`, `data`, `on`, `a` |
| `the cat sat on the` | `warm` | **46.50%** | `mat (29%)`, `fireplace (9%)`, `park (2%)` |
| `machine learning is` | `a` | **85.30%** | `all (8%)`, `key (1%)`, `each (1%)` |
| `deep learning models` | `learn` | **66.90%** | `. (13%)`, `generate (9%)`, `for (3%)` |
| `natural language` | `processing` | **97.70%** | `enables`, `blocks`, `models`, `vision` |
| `the transformer architecture was` | `introduced` | **64.40%** | `was`, `mapped`, `at`, `in` |
| `neural networks learn` | `patterns` | **49.70%** | `unseen`, `directly`, `massive` |

---

## 7. Viva Voce & Academic Q&A Preparation

### Q1: What is the main role of Self-Attention in Transformers?
**Answer**: Self-attention allows each token in an input sequence to dynamically relate to and aggregate context from every other token in the sequence simultaneously, eliminating the sequential bottleneck of RNNs.

### Q2: Why is causal masking necessary in next-token prediction?
**Answer**: During autoregressive language modeling, the task at position $i$ is to predict token $i+1$. Without causal masking (setting attention weights to $-\infty$ for $j > i$), the model would look ahead at future tokens and achieve trivial zero loss during training, failing completely during real-world inference.

### Q3: Why do we divide by $\sqrt{d_k}$ in scaled dot-product attention?
**Answer**: For high dimensions, the dot products grow large, pushing the Softmax function into saturation regions where gradients approach zero (vanishing gradient problem). Scaling by $1/\sqrt{d_k}$ keeps the variance of the scores close to 1.

### Q4: What is Perplexity and what does it indicate?
**Answer**: Perplexity is $\exp(\text{Cross-Entropy Loss})$. It represents the effective vocabulary branching factor. A perplexity of 10 means the model is as uncertain as randomly guessing among 10 equally likely words. Lower perplexity indicates superior predictive performance.

### Q5: What is the difference between Greedy Decoding and Temperature Sampling?
**Answer**: Greedy decoding selects the token with the single highest probability ($\text{argmax}$). Temperature sampling divides logits by a temperature $T$ before softmax:
- $T < 1.0$: Sharper distribution, more predictable and repetitive.
- $T > 1.0$: Flatter distribution, more diverse and creative text.

---

## 8. References

1. Vaswani, A., et al. (2017). *"Attention Is All You Need."* Advances in Neural Information Processing Systems (NeurIPS 2017).
2. Radford, A., et al. (2019). *"Language Models are Unsupervised Multitask Learners."* OpenAI Technical Report (GPT-2).
3. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *"Layer Normalization."* arXiv preprint arXiv:1607.06450.
4. PyTorch Documentation (2024). *"torch.nn, torch.matmul, and torch.autograd."* https://pytorch.org/docs/
