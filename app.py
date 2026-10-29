import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from src.tokenizer import SimpleTokenizer
from src.transformer import TransformerLM
from src.generate import (
    load_model_and_tokenizer,
    predict_next_token_candidates,
    generate_text,
)

# Page configuration
st.set_page_config(
    page_title="Transformer Next-Token Predictor",
    page_icon="🧠",
    layout="wide",
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pt")
VOCAB_PATH = os.path.join(BASE_DIR, "models", "vocab.json")
HISTORY_PATH = os.path.join(BASE_DIR, "models", "training_history.json")
LOSS_PLOT_PATH = os.path.join(BASE_DIR, "models", "training_loss.png")


@st.cache_resource
def get_model_and_tokenizer():
    """Caches model and tokenizer in memory for instant responses."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        return None, None
    return load_model_and_tokenizer(MODEL_PATH, VOCAB_PATH, device="cpu")


# Title and Header
st.title("🧠 Transformer-Based Next-Token Prediction Model")
st.caption(
    "Academic Mini-Project: Built from scratch with PyTorch using Basic Scaled Dot-Product Self-Attention"
)

# Sidebar: Configuration and Project Info
with st.sidebar:
    st.header("⚙️ Model Controls")
    temperature = st.slider("Temperature (Creativity)", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Generation Tokens", min_value=1, max_value=30, value=12, step=1)
    sampling_mode = st.radio("Decoding Strategy", ["Greedy Decoding (Deterministic)", "Temperature Sampling (Creative)"])
    use_sample = sampling_mode == "Temperature Sampling (Creative)"

    st.markdown("---")
    st.header("📌 Architecture Specs")
    st.markdown(
        """
        - **Model**: Autoregressive Decoder
        - **Embedding Dim ($d_{\\text{model}}$)**: 64
        - **Attention Heads**: 4
        - **Transformer Layers**: 2
        - **Context Window**: 16 tokens
        - **Attention**: Scaled Dot-Product with Causal Mask
        """
    )

model, tokenizer = get_model_and_tokenizer()

if model is None or tokenizer is None:
    st.error(
        "⚠️ Trained model weights (`models/best_model.pt`) or vocabulary (`models/vocab.json`) were not found.\n\n"
        "**To train the model on your CPU:**\n"
        "```powershell\n"
        "python src/train.py --epochs 30\n"
        "```\n"
        "Once training completes, refresh this page."
    )
    st.stop()

# Load training history if available
history_data = None
if os.path.exists(HISTORY_PATH):
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            history_data = json.load(f)
    except Exception:
        pass

# Navigation Tabs
tab_predict, tab_generate, tab_attention, tab_metrics = st.tabs([
    "🎯 Next-Token Prediction",
    "✍️ Text Generation",
    "🔍 Self-Attention Visualizer",
    "📊 Training & Evaluation",
])

# ----------------- TAB 1: NEXT-TOKEN PREDICTION -----------------
with tab_predict:
    st.subheader("Interactive Next-Token Predictor")
    st.markdown("Enter a short prefix phrase to view the model's highest-probability next token and top-5 candidates.")

    default_prompt = "machine learning is"
    user_prompt = st.text_input("Enter Prompt:", value=default_prompt, key="next_token_input")

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        predict_btn = st.button("Predict Next Token", type="primary")

    if predict_btn or user_prompt:
        candidates = predict_next_token_candidates(model, tokenizer, user_prompt, top_k=5)

        if candidates:
            top_candidate = candidates[0]
            st.success(
                f"**Input:** `{user_prompt}`  ➡️  **Predicted Next Token:** `{top_candidate['token']}` "
                f"(Probability: **{top_candidate['percentage']}**)"
            )

            col1, col2 = st.columns([1.2, 1])

            with col1:
                st.markdown("##### 🏆 Top-5 Candidate Tokens")
                tokens = [c["token"] for c in candidates]
                probs = [c["probability"] * 100 for c in candidates]

                fig, ax = plt.subplots(figsize=(6, 3.2))
                bars = ax.barh(tokens[::-1], probs[::-1], color="#2b5c8f")
                ax.set_xlabel("Probability (%)", fontsize=10)
                ax.set_xlim(0, max(probs) * 1.25 + 5)
                for bar in bars:
                    w = bar.get_width()
                    ax.annotate(
                        f"{w:.1f}%",
                        xy=(w, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=9,
                    )
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("##### 📋 Probability Breakdown")
                st.table([
                    {
                        "Rank": i + 1,
                        "Token": c["token"],
                        "Token ID": c["token_id"],
                        "Probability": c["percentage"],
                    }
                    for i, c in enumerate(candidates)
                ])

# ----------------- TAB 2: TEXT GENERATION -----------------
with tab_generate:
    st.subheader("Autoregressive Text Generation")
    st.markdown("Generate coherent sentences one token at a time starting from your initial prompt.")

    gen_prompt = st.text_input("Enter Starting Text:", value="the transformer architecture was", key="gen_input")
    generate_btn = st.button("Generate Text Continuation", type="primary")

    if generate_btn:
        with st.spinner("Generating autoregressively with self-attention..."):
            completed_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=gen_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                sample=use_sample,
                repetition_penalty=1.2,
            )

        st.markdown("#### Generated Output:")
        st.info(f"**Result:** {completed_text}")
        st.markdown(
            f"*Decoding Mode:* `{sampling_mode}` | *Temperature:* `{temperature}` | *Max Tokens:* `{max_tokens}`"
        )

# ----------------- TAB 3: SELF-ATTENTION VISUALIZER -----------------
with tab_attention:
    st.subheader("Causal Self-Attention Weight Inspection")
    st.markdown(
        "Inspect how each token attends to prior tokens. "
        "Notice that the upper triangle is strictly **0.0** due to the causal mask!"
    )

    attn_prompt = st.text_input("Analyze phrase:", value="neural networks learn patterns", key="attn_input")

    if attn_prompt:
        token_ids = tokenizer.encode(attn_prompt)
        tokens_words = [tokenizer.idx2word.get(tid, "<UNK>") for tid in token_ids]

        if len(token_ids) > 1:
            inp = torch.tensor([token_ids], dtype=torch.long)
            with torch.no_grad():
                _, attn_layers = model(inp, causal=True, return_attention=True)

            # Show Layer 0, Head 0 attention matrix
            attn_matrix = attn_layers[0][0, 0].cpu().numpy()

            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            cax = ax.matshow(attn_matrix, cmap="Blues")
            plt.colorbar(cax, fraction=0.046, pad=0.04)

            ax.set_xticks(range(len(tokens_words)))
            ax.set_yticks(range(len(tokens_words)))
            ax.set_xticklabels(tokens_words, rotation=45, ha="left", fontsize=9)
            ax.set_yticklabels(tokens_words, fontsize=9)
            ax.set_title("Self-Attention Weights (Layer 1, Head 1)", pad=20, fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.caption(
                "💡 **Causal Masking Proof:** Rows represent query positions; columns represent key positions. "
                "Any position $(i, j)$ where $j > i$ has attention weight zero, preventing cheating from future words."
            )

# ----------------- TAB 4: TRAINING & EVALUATION METRICS -----------------
with tab_metrics:
    st.subheader("Model Performance & Training Curves")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Parameters", f"{model.count_parameters():,}")
    with m_col2:
        st.metric("Vocabulary Size", f"{tokenizer.vocab_size} tokens")
    with m_col3:
        if history_data:
            st.metric("Best Val Loss", f"{min(history_data['val_loss']):.3f}")
        else:
            st.metric("Best Val Loss", "Available after run")
    with m_col4:
        if history_data:
            st.metric("Final Perplexity", f"{history_data['perplexity'][-1]:.2f}")
        else:
            st.metric("Final Perplexity", "N/A")

    st.markdown("---")

    if os.path.exists(LOSS_PLOT_PATH):
        st.image(LOSS_PLOT_PATH, caption="Cross-Entropy Training vs. Validation Loss across Epochs", use_column_width=True)
    elif history_data:
        epochs = range(1, len(history_data["train_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, history_data["train_loss"], label="Train Loss", marker="o")
        ax.plot(epochs, history_data["val_loss"], label="Validation Loss", marker="s", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross Entropy Loss")
        ax.legend()
        ax.grid(True, linestyle=":")
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Train the model using `python src/train.py` to generate loss plots.")
